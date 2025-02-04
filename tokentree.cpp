#include "tokentree.h"
#include <set>

void LLMBuffer::init()
{
	/* init callbacks */
	notify_invalidate = [](int,int) {};
	notify_new_logit = [](int,int,float) {};
	notify_new_predictions = []() {};
	notify_change_tail = [](int, std::string) {};
	
	/* init llama.cpp */
	common_init();
	
	common_params params;
	
	params.n_gpu_layers = 99;

	ggml_numa_strategy st;
	
	llama_backend_init();
    llama_numa_init(params.numa);
	
	// prepare work thread
	work_batch = llama_batch_init(512, 0, 1);
	work_base = &root;
	is_working = false;
	wq_head_invalid = false;
	//work_done.connect(sigc::mem_fun(this,&LLMBuffer::on_work_done));
	work_done_flag = false;

	load_model("Qwen2.5-3B.Q4_K_M.gguf");
	//load_model("Phi-3.5-mini-instruct-Q4_K_M.gguf");
}

void LLMBuffer::load_model(const char *fn)
{
	llama_model_params model_params = llama_model_default_params();

	model_params.n_gpu_layers = 99; // offload all layers to the GPU
	llama_model *new_model = llama_load_model_from_file(fn, model_params);

	if (new_model == NULL) {
		fprintf(stderr , "%s: error: unable to load model\n" , __func__);
		return;
	}

	std::string text;

	// load succeeded, replace our model
	if(model) {
		// save text
		text = render(&root);

		llama_model_free(model);
	}
	model = new_model;

	// gather basic metadata
	model_fn = fn;
	for(int i=0; i<llama_model_meta_count(model); i++) {
		char k_buf[256], v_buf[256];
		llama_model_meta_key_by_index(model, i, k_buf, 256);
		if(!strcmp(k_buf, "general.architecture")) {
			llama_model_meta_val_str_by_index(model, i, v_buf, 256);
			model_arch = v_buf;
		} else if(!strcmp(k_buf, "general.size_label")) {
			llama_model_meta_val_str_by_index(model, i, v_buf, 256);
			model_size = v_buf;
		}
	}
	

	vocab = llama_model_get_vocab(model);

	// initialize the context
	ctx_params = llama_context_default_params();

	ctx_params.n_ctx = 2048;

	if(ctx) llama_free(ctx); //free old context?
	ctx = llama_new_context_with_model(model, ctx_params);

	if (ctx == NULL) {
		fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
		exit(1);
	}

	n_vocab = llama_vocab_n_tokens(vocab);

	/* init root token */
	root.clear_children();
	root.base_pos=0;
	root.depth=0;
	root.is_accepted=true;
	root.set_tok(llama_vocab_bos(vocab));
	root.str="";
	root.parent=NULL;
	root.sel=0;
	root.has_logit=false;
	root.ctx_snapshot = std::shared_ptr<uint8_t[]>(new uint8_t[llama_get_state_size(ctx)]);
	llama_copy_state_data(ctx,root.ctx_snapshot.get());
	ctx_state=NULL;

	rebuild(&root, text, text.size());
}

void LLMBuffer::insert(int pos, std::string text)
{
	TTE *start = pos2wordent(pos);
	std::string tail = render(start);
	printf("pre-ins: '%s', '%s', %d\n", tail.c_str(), text.c_str(), pos - start->base_pos);
	tail.insert(pos - start->base_pos, text);
	printf("post-ins: '%s'\n", tail.c_str());
	
	rebuild(start, tail, pos+text.size(), text.size());
}

void LLMBuffer::erase(int from, int to)
{
	TTE *start = pos2wordent(from);
	std::string tail = render(start);
	tail.erase(from - start->base_pos, to - from);
	
	rebuild(start, tail, from, from - to);
}

/* given a buffer offset, get pointer to live token tree entry covering that offset */
TTE *LLMBuffer::pos2ent(int pos)
{
	int offs=0;
	TTE *cur = &root;
	
	while(cur == &root || offs<pos || !cur->str_size) {
		offs+=cur->str_size;
		
		if(!cur->children.size()) return cur;
		
		cur=cur->children[cur->sel];
	}
	
	return cur->parent;
}

/* given a buffer offset, get pointer to live token tree entry covering the first word before that offset */
TTE *LLMBuffer::pos2wordent(int pos)
{
	int offs=0;
	TTE *cur = &root;
	
	while(offs<pos) {
		offs+=cur->str_size; 
		
		if(!cur->children.size()) goto pos2wordent_end;
		
		cur=cur->children[cur->sel];
	}
	
	if(cur->parent) cur=cur->parent;
	
	while(cur->parent && cur->str.find(' ')==std::string::npos) {
		cur=cur->parent;
pos2wordent_end:;
//		printf("back: %lX, '%s'\n", cur->parent, cur->str.c_str());
	}
	
	return cur;
}

/* convert live path from token tree entry to string */
std::string LLMBuffer::render(TTE *tt, int max_tok, bool render_predictions)
{
	std::string ret;
	while(tt && max_tok && (render_predictions || tt->is_accepted)) {
		ret += tt->str;
		tt = ((tt->children.size()>0)&&(tt->sel>=0))?(TTE*)tt->children[tt->sel]:NULL;
		--max_tok;
	}
	return ret;
}

void LLMBuffer::rebuild(TTE *start, std::string text, int change_end, int reconcile_offset)
{
	const int n_tokens = -llama_tokenize(vocab, text.c_str(), text.size(), NULL, 0, start->tok==llama_vocab_bos(vocab), true);
	std::vector<llama_token> tokens_list(n_tokens);
	llama_tokenize(vocab, text.c_str(), text.size(), tokens_list.data(), tokens_list.size(), start->tok==llama_vocab_bos(vocab), true);
	
	if(start->tok==llama_vocab_bos(vocab) && (tokens_list.size()>=1 && tokens_list[0]!=llama_vocab_bos(vocab))) {
		tokens_list.insert(tokens_list.begin(),llama_vocab_bos(vocab));
	}
	
	purgeWork(start->depth);
	notify_invalidate(start->base_pos, start->base_pos + text.size());
	
	if(!tokens_list.size()) {
		// complete deletion
		if(start->parent) {
			for(auto i = start->parent->children.begin(); i!=start->parent->children.end(); ++i) {
				if((*i)==start) {
					start->parent->children.erase(i);
					delete *i;
					return;
				}
			}
		}
	}
	
	TTE *rebuild_root = start->parent?start->parent:start;
	
	printf("rebuild '%s'\n", text.c_str());


	TTE *target_p=rebuild_root, *old_p=start;
	size_t source_i=0;

	// skip matching tokens in the beginning
	while (source_i < tokens_list.size() && old_p && tokens_list[source_i] == old_p->tok) {
		++source_i;
		target_p = old_p;
		if(old_p->children.size()) {
			old_p = old_p->children[target_p->sel];

			if(!old_p->is_accepted) {
				delete old_p;
				old_p = NULL;
				break;
			}
		} else old_p = NULL;
	}

	// insert new tokens
	while (source_i < tokens_list.size()) {
		int next_basepos = target_p->base_pos + target_p->str_size;

		if(old_p) printf("try reconcile: %d @%d with old %d '%s' @%d\n", tokens_list[source_i], next_basepos, old_p->tok, old_p->str.c_str(), old_p->base_pos + reconcile_offset);

		// if we have already added all the new characters, and the tokenisaton has realigned,
		// then we can hook in the rest of the old tree
		if(old_p && old_p->tok == tokens_list[source_i] && next_basepos >= change_end && old_p->base_pos + reconcile_offset == next_basepos) {
			if(target_p->children.size()) target_p->children[target_p->sel] = old_p;
			else {
				target_p->children.push_back(old_p);
				target_p->sel = 0;
			}
			old_p->parent = target_p;
			old_p->reroot(1+target_p->depth-old_p->depth, reconcile_offset);
			goto rebuild_linked;
		}

		TTE *next = new TTE(this);

		if(target_p->children.size()) target_p->children[target_p->sel] = next;
		else {
			target_p->children.push_back(next);
			target_p->sel = 0;
		}

		next->base_pos = next_basepos;
		next->depth = target_p->depth + 1;
		next->parent = target_p;
		next->is_accepted = true;
		next->sel = 0;
		next->has_logit = false;
		next->set_tok (tokens_list[source_i]);
		++source_i;
		target_p = next;

		printf("%d '%s' . ",target_p->tok, target_p->str.c_str());

		// advance old_p
		while(old_p && old_p->base_pos + reconcile_offset < target_p->base_pos + target_p->str_size) {
			if(old_p->children.size()) {
				TTE *oldold_p = old_p;
				old_p = old_p->children[old_p->sel];
				// delete skipped token, except for the one child we keep
				oldold_p->children[oldold_p->sel] = *oldold_p->children.rbegin();
				oldold_p->children.pop_back();
				delete oldold_p;
				// if we entered prediction territory, delete that too
				if(!old_p->is_accepted) {
					delete old_p;
					old_p = NULL;
				}
			} else {
				delete old_p;
				old_p = NULL;
			}
		}
	}
	printf("\n");

	// if we are here, we deposited the entire new token string. No predictions etc. should be allowed to live after it
	target_p->clear_children();
rebuild_linked:;
	
	enqueueWork(WL_SCORE, rebuild_root);
}

bool validate_utf8(const char *str, int len) {
    int n;
    for (int i = 0; i < len; ++i) {
        unsigned char c = (unsigned char) str[i];
        if (0x00 <= c && c <= 0x7f) {
            n=0; // 0bbbbbbb
        } else if ((c & 0xE0) == 0xC0) {
            n=1; // 110bbbbb
        } else if ( c==0xed && i<(len-1) && ((unsigned char)str[i+1] & 0xa0)==0xa0) {
            return false; //U+d800 to U+dfff
        } else if ((c & 0xF0) == 0xE0) {
            n=2; // 1110bbbb
        } else if ((c & 0xF8) == 0xF0) {
            n=3; // 11110bbb
        } else {
            return false;
        }

        for (int j = 0; j < n && i < len; ++j) { // n bytes matching 10bbbbbb?
            if ((++i == len) || (( (unsigned char)str[i] & 0xC0) != 0x80)) {
                return false;
            }
        }
    }
    return true;
}

void LLMBuffer::actualize(TTE *start)
{
	std::string txt = render(start);
	/* leap over token to get valid UTF-8 */
	if(!validate_utf8(txt.c_str(),txt.length())) {
		printf("utf-8 leap\n");
		TTE *pos = start;
		while(pos->children.size()>0 && pos->children[pos->sel]->is_accepted) pos=pos->children[pos->sel];
		
		auto extend = [&pos,&txt,&start,this](){ 
			if(pos->children.size()>0 && !pos->children[pos->sel]->is_accepted) {
				pos->children[pos->sel]->is_accepted=true;
				pos=pos->children[pos->sel];
				txt = render(start);
				return true;
			}
			return false;
		};
			
		while(extend() && !validate_utf8(txt.c_str(),txt.length()));
		if(!validate_utf8(txt.c_str(),txt.length())) {
			start->is_accepted=false;
			txt="";
		}
	}
	printf("actualize %lX, %d, '%s'\n", start, start->base_pos, txt.c_str());
	notify_change_tail(start->base_pos, txt);
	while(start && start->is_accepted)
	{
		if(start->has_logit)
			notify_new_logit(start->base_pos, start->base_pos+start->str_size, start->logit-start->max_logit);
			
		if(start->children.size()>start->sel) start=start->children[start->sel];
		else start=NULL;
	}
}

/* add workload to be executed after everything else */
void LLMBuffer::enqueueWork(workload_type t, TTE *target, int gen_extra)
{
	printf("enqueue from '%s'@%d (+%d)\n", target->str.c_str(), target->depth, target->base_pos);
	wq.push_back( TTWorkload { t, target->base_pos, target->base_pos + target->str_size, target->depth, target, gen_extra } );
	
	try_start_working();
}

/* add workload to be executed ASAP */
void LLMBuffer::injectWork(workload_type t, TTE *target, int gen_extra)
{
	//printf("inject from '%s'\n", target->str.c_str());
	
	if(!is_working) wq.push_front( TTWorkload { t, target->base_pos, target->base_pos + target->str_size, target->depth, target, gen_extra } );
	else wq.insert( ++wq.begin(), TTWorkload { t, target->base_pos, target->base_pos + target->str_size, target->depth, target, gen_extra } );
	
	try_start_working();
}

void LLMBuffer::purgeWork(int start_depth)
{
	if(!wq.size()) return;
	if(wq.front().depth >= start_depth) {
		wq_head_invalid = true;
		printf("purge '%s'\n", wq.front().target->str.c_str());
	}
	for(auto i = ++wq.begin(), j=i; i!=wq.end(); ) {
		++j;
		
		if(i->depth >= start_depth) {
			printf("purge '%s'\n", i->target->str.c_str());
			wq.erase(i);
		}
		
		i=j;
	}
}

void LLMBuffer::purgePredictionWork()
{
	//printf("purge pw!\n");
	if(!wq.size()) return;
	if(wq.front().wl_type == WL_BRANCH || wq.front().wl_type == WL_PREDICT) {
		//wq_head_invalid = true;
	}
	for(auto i = ++wq.begin(), j=i; i!=wq.end(); ) {
		++j;
		
		if(i->wl_type == WL_BRANCH || i->wl_type == WL_PREDICT) {
			wq.erase(i);
		}
		
		i=j;
	}
}

void LLMBuffer::CheckWork()
{
	if(work_done_flag) {
		work_done_flag = false;
		on_work_done();
	}
}

void LLMBuffer::on_work_done()
{
	is_working=false;
	
	if(!wq_head_invalid) {
		std::shared_ptr<uint8_t[]> snap;
		if(llm_state_changed && ((work_base->depth%snapshot_freq)+work_batch.n_tokens)>=snapshot_freq)  {
			snap = std::shared_ptr<uint8_t[]>(new uint8_t[llama_get_state_size(ctx)]);
			size_t copied = llama_copy_state_data(ctx,snap.get());
			printf("snap (%zu/%zu bytes), as work base is at %d and processed %d extra tokens.\n", copied, llama_get_state_size(ctx), work_base->depth, work_batch.n_tokens);
		}
		
		switch(wq.front().wl_type) {
		case WL_SCORE: {
			TTE *t = wq.front().target;
			int gen_extra = wq.front().gen_extra;

			/* TODO make sure it's okay to skip if even just the selected child has a logit already */
			if(!t->children.size() || t->children[t->sel]->has_logit) {
				// if we are here, work_batch is NOT valid and we should not read logits
				wq.pop_front();
				wq_head_invalid=false;

				printf("score advance: %p @%d\n", t, t->depth);

				// cross all already-scored children at once to avoid huge call stacks
				while(t->children.size() && t->children[t->sel]->has_logit) t = t->children[t->sel];
				if(t->children.size()) //!t->children[t->sel]->has_logit
					injectWork(WL_SCORE, t, gen_extra);

				break;
			}

			float *logits = llama_get_logits_ith(ctx, work_batch.n_tokens - 1);
			float max_logit = *std::max_element(logits, logits+n_vocab);
						
			wq.pop_front();
			wq_head_invalid=false;
			
			// render any new logits we generated
			renderLogitsFromBatch(work_base, work_batch.n_tokens-1, &work_batch);
			
			printf("decoded '%s' from %d tokens.\n", t->str.c_str(), work_batch.n_tokens);
			
			for(int i=0;i<t->children.size();++i) {
				auto &tt = *t->children[i];
				if (!tt.has_logit) {
					tt.logit = logits[tt.tok];
					tt.max_logit = max_logit;
					tt.has_logit = true;
					tt.ctx_snapshot = snap;
			
					printf("'%s' (%d) at %d get new logit %.2f\n", tt.str.c_str(), tt.tok, tt.depth, tt.logit);
					fflush(stdout);
					
					if(t->sel == i && tt.is_accepted) {
						notify_new_logit(tt.base_pos, tt.base_pos+tt.str_size, tt.logit - tt.max_logit);
						
						injectWork(WL_SCORE, &tt, gen_extra);
					} 
				}
			}
			break;
			}
		case WL_PREDICT: {
			TTE *t = wq.front().target;
			int gen_extra = wq.front().gen_extra;
			
			wq.pop_front();
			wq_head_invalid=false;
			
			// if we are just predicting and there is already a prediction, advance quietly
			if(t->children.size()>0) {
				if(gen_extra>0)
					injectWork(WL_PREDICT, t->children[t->sel], gen_extra-1);
				try_start_working();
				return;
			}
			
			// render any new logits we generated
			renderLogitsFromBatch(work_base, work_batch.n_tokens-1, &work_batch);
			
			float *logits = llama_get_logits_ith(ctx, work_batch.n_tokens - 1);
			
			float l_max=-999.9; int i_max=0;
			for(int i=0;i<n_vocab;++i) {
				if(logits[i]>l_max) {
					l_max = logits[i];
					i_max = i;
				}
			}
			
			t->children.push_back(new TTE(this));
			t->sel=0;
			TTE *next = t->children[0];
			next->base_pos = t->base_pos + t->str_size;
			next->depth = t->depth + 1;
			next->parent = t;
			next->is_accepted = false;
			next->set_tok(i_max);
			next->logit = l_max;
			next->max_logit = l_max;
			next->has_logit = true;
			next->sel = 0;
			next->ctx_snapshot = snap;
			
			printf("new pred: '%s' (%d) at %d with logit %.2f\n", next->str.c_str(), next->tok, next->depth, next->logit);
			/* if(next->tok == 362) {
				printf("!?\n");
			} */ // "What is happening here? A"
			
			notify_new_predictions();
			
			if(gen_extra>0)
				injectWork(WL_PREDICT, next, gen_extra-1);
			try_start_working();
			
			break;
			}
		case WL_BRANCH: {
			TTE *t = wq.front().target;
			int gen_extra = wq.front().gen_extra;
			
			wq.pop_front();
			wq_head_invalid=false;
			
			// if there is already a bottom alternative, advance quietly
			if(t->children.size() > (t->sel+1)) {
				if(gen_extra>0) {
					if(t->sel>0) injectWork(WL_PREDICT, t->children[t->sel-1], gen_extra-1);
					injectWork(WL_PREDICT, t->children[t->sel+1], gen_extra-1);
					injectWork(WL_PREDICT, t->children[t->sel], predict_main-1);
				}
				try_start_working();
				return;
			}
			
			// render any new logits we generated
			renderLogitsFromBatch(work_base, work_batch.n_tokens-1, &work_batch);
			
			float *logits = llama_get_logits_ith(ctx, work_batch.n_tokens - 1);
			float max_logit = *std::max_element(logits, logits+n_vocab);
			
			std::set<int> exclude;
			for(TTE *c : t->children) exclude.insert(c->tok);
			
			while(t->children.size() <= (t->sel+1)) {
				float l_max=-999.9; int i_max=0;
				for(int i=0;i<n_vocab;++i) {
					if(exclude.count(i)) continue;
					if(logits[i]>l_max) {
						l_max = logits[i];
						i_max = i;
					}
				}
				
				t->children.push_back(new TTE(this));
				TTE *next = t->children[t->children.size()-1];
				next->base_pos = t->base_pos + t->str_size;
				next->depth = t->depth + 1;
				next->parent = t;
				next->is_accepted = false;
				next->set_tok(i_max);
				next->logit = l_max;
				next->max_logit = max_logit;
				next->has_logit = true;
				next->sel = 0;
				next->ctx_snapshot = snap;
				
				printf("new branch: '%s' (%d) at %d with logit %.2f\n", next->str.c_str(), next->tok, next->depth, next->logit);
				/*if(next->tok == 3555) {
					printf("!?\n");
				}*/ // "What is happening here? What"
				
				exclude.insert(i_max);
			}
			
			notify_new_predictions();
			
			if(gen_extra>0) {
				injectWork(WL_PREDICT, t->children[t->sel], predict_main-1);
				enqueueWork(WL_PREDICT, t->children[t->sel+1], gen_extra-1);
				if(t->sel>0) enqueueWork(WL_PREDICT, t->children[t->sel-1], gen_extra-1);
				//injectWork(WL_PREDICT, t->children[t->sel], gen_extra-1);
			}
			try_start_working();
			
			break;
			}
		}
		
	} else {
		wq.pop_front();
		wq_head_invalid=false;
	}
	
	try_start_working();
}

void LLMBuffer::try_start_working()
{
	if(is_working) return;
	
	llm_state_changed = false;
	
	if(wq_head_invalid && wq.size()) {
		wq.pop_front();
		wq_head_invalid = false;
	}
			
	if(wq.size()) {
		// may not need to rerun the LLM if we are in predict mode
		auto &wl = wq.front();
		
		if(    ( wl.wl_type == WL_PREDICT && wl.target->children.size()>0 )
			|| ( wl.wl_type == WL_BRANCH  && wl.target->children.size()>(wq.front().target->sel+1) )
			|| ( wl.wl_type == WL_SCORE   && (wl.target->children.size()==0 || wl.target->children[wl.target->sel]->has_logit) ) )
		{
			on_work_done();
			return;
		}
		
		//printf("making batch for: type %d, target: '%s' (%d) at %d (+%d)\n", wl.wl_type, wl.target->str.c_str(), wl.target->tok, wl.target->depth, wl.target->base_pos);
		std::shared_ptr<uint8_t[]> p = prepareBatch(&wq.front());

		/*if(!work_batch.n_tokens) {
			//empty batch??
			wq.pop_front();
			try_start_working();
			return;
		}*/

		is_working = true;
		wthread = new std::thread(
		  [this,p]
		  {
			if(p) llama_set_state_data(ctx,p.get());
			llama_decode(ctx, work_batch);
			llm_state_changed = true;
			//work_done.emit();
			work_done_flag = true;
		  });
	}
}

void LLMBuffer::renderLogitsFromBatch(TTE* t, int n, llama_batch *b)
{
	int i=0;
	while(n) {
		if(t->children.size()) {
			TTE *tt = t->children[t->sel];
			if(b->logits[i]) {
				float *logits = llama_get_logits_ith(ctx, i);
				float max_logit = *std::max_element(logits, logits+n_vocab);
				
				tt->logit = logits[tt->tok];
				tt->max_logit = max_logit;
				tt->has_logit = true;
				
				if(tt->is_accepted) {
					printf("'%s' (%d) len=%d at %d batch new logit %.2f\n", tt->str.c_str(), tt->tok, tt->str_size, tt->depth, tt->logit);
					
					notify_new_logit(tt->base_pos, tt->base_pos+tt->str_size, tt->logit - tt->max_logit);
				}
			}
			t=tt;
		} else break;
		--n; ++i;
	}
}

std::shared_ptr<uint8_t[]> LLMBuffer::prepareBatch(TTWorkload *wl)
{
	common_batch_clear(work_batch);
	if(ctx_state && ctx_state == wl->target->parent) {
		common_batch_add(work_batch, wl->target->tok, wl->target->depth, { 0 }, false);
		work_batch.logits[0]=true;
		work_base = wl->target;
		ctx_state = wl->target;
		printf("single step by '%s' (%d) at %d (+%d)\n", wl->target->str.c_str(), wl->target->tok, wl->target->depth, wl->target->base_pos);

		return nullptr;
	} else {
		// need to find context snapshot and advance from it
		std::vector<llama_token> toks;
		std::vector<bool> need_logits;
		need_logits.push_back(true);
		
		TTE *pos = wl->target;
		while(!pos->ctx_snapshot){
			toks.push_back(pos->tok);
			if(!pos->has_logit) need_logits.push_back(true);
			else need_logits.push_back(false);
			
			pos = pos->parent;
		}
		toks.push_back(pos->tok); // assume snapshot was made BEFORE this token
		
		// add tokens we saw in reverse order
		int d = pos->depth;
		for(int i = toks.size()-1; i>=0; --i) {
			common_batch_add(work_batch, toks[i], d++, { 0 }, false);
			work_batch.logits[toks.size()-1-i] = need_logits[i];
		}
		
		// reset context to snapshot
		//llama_free(ctx);
		//ctx = llama_new_context_with_model(model, ctx_params);
		//llama_reinit_kv_cache(ctx, ctx_params);
		//int n_copied = llama_set_state_data(ctx,pos->ctx_snapshot.get());
		
		// debug output
		//std::string txt;
		char txt[1024], *p=txt;
		for(int i = toks.size()-1; i>=0; --i) {
			//txt+=llama_token_to_piece(ctx,toks[i],false);
			p += llama_token_to_piece(vocab, toks[i], p, 1024-(p-txt), 0, false);
			//printf("%d ",toks[i]);
		}
		*p = 0;
		printf("reset to '%s' (%d) at %d (+%d), catchup '%s'\n", pos->str.c_str(), pos->tok, pos->depth, pos->base_pos, txt);
		
		work_base = pos;
		ctx_state = wl->target;

		return pos->ctx_snapshot;
	}
}

void LLMBuffer::req_alts_at_pos(int pos)
{
	TTE *cur = pos2ent(pos);
	printf("req alts from '%s' (%d) at %d (+%d)\n", cur->str.c_str(), cur->tok, cur->depth, cur->base_pos);
	purgePredictionWork(); // get rid of old prediction tasks
	injectWork(WL_BRANCH, cur, predict_alt);
	injectWork(WL_PREDICT, cur, predict_main);
	try_start_working();
}

void LLMBuffer::get_alts_at_pos(int pos, std::string &above, std::string &selected, std::string &below, int &delta)
{
	TTE *cur = pos2ent(pos);
	delta = cur->base_pos + cur->str_size - pos;
	if(cur->sel>0) 
		above = render(cur->children[cur->sel-1],4,true);
	else above = "";
	if(cur->children.size()>(cur->sel))
		selected = render(cur->children[cur->sel],4,true);
	else selected = "";
	if(cur->children.size()>(cur->sel+1)) {
		below = render(cur->children[cur->sel+1],4,true);
	} else {
		below = "";
	}
}

void LLMBuffer::alt_next(int pos)
{
	TTE *cur = pos2ent(pos);
	if(cur->children.size()>(cur->sel+1))
		++cur->sel;
		
	if(cur->children.size()) {
		actualize(cur->children[cur->sel]);
		enqueueWork(WL_SCORE, cur->children[cur->sel]); // previous changes might have left this branch unscored
	}
	
	injectWork(WL_BRANCH, cur, predict_alt);
	try_start_working();
}

void LLMBuffer::alt_prev(int pos)
{
	TTE *cur = pos2ent(pos);
	if(cur->sel>0)
		--cur->sel;
	
	if(cur->children.size()) {
		actualize(cur->children[cur->sel]);
		enqueueWork(WL_SCORE, cur->children[cur->sel]); // previous changes might have left this branch unscored
	}
	
	injectWork(WL_BRANCH, cur, predict_alt);
	try_start_working();
}

int LLMBuffer::alt_commit(int pos)
{
	TTE *cur = pos2ent(pos);
	if(cur->children.size()) {
		cur->children[cur->sel]->is_accepted = true;
		actualize(cur->children[cur->sel]);
		// skip to not end in the middle of a UTF-8 codon
		cur = cur->children[cur->sel];
		int posn;
		do {
			posn = cur->base_pos + cur->str_size;
			if(cur->children.size()) {
				cur = cur->children[cur->sel];
				if(!cur->is_accepted) cur=NULL;
			} else cur = NULL;
		} while(cur && (cur->str[0]&0xC0) == 0x80);
		return posn;
	} else return cur->base_pos + cur->str_size;
}

int LLMBuffer::alt_back(int pos)
{
	TTE *cur = pos2ent(pos);
	
	if(cur->base_pos == pos) {
		if(cur->parent) cur = cur->parent;
	}
	int posn = cur->base_pos;
	//skip to not end in the middle of a UTF-8 codon
	while(cur && (cur->str[0]&0xC0) == 0x80) {
		cur = cur->parent;
		if(cur) posn = cur->base_pos;
	}
	return posn;
}

void LLMBuffer::debug_tte(TTE *pos)
{
	printf("tok '%s' at %d (%lX): parent = %lX, children = [ ", pos->str.c_str(), pos->base_pos, pos, pos->parent);
	for(auto &a : pos->children) {
		printf("%lX ",&a);
	}
	printf("]\n");
	for(auto &a : pos->children) {
		debug_tte(a);
	}
}

void TTE::set_tok(llama_token t)
{
	//if(tok!=t) ctx_snapshot.reset(); // snapshot was invalidated, reset
	
	tok = t;
	char buf[128];
	str_size = llama_token_to_piece(buffer->vocab, t, buf, 128, 0, depth>0);
	//str_size = llama_detokenize(buffer->vocab, &t, 1, buf, 128, false, false);
	buf[str_size]=0;
	str = buf;

	/*
	if(str.validate()) str_size = str.size();
	else {
		if((((unsigned char)str.c_str()[0])&0xC0) != 0x80) str_size = str.size(); // broken token at the end should register as 1
		else str_size=0;
	}
	*/
}

void TTE::reroot(int delta_depth, int delta_pos)
{
	has_logit = false;
	depth += delta_depth;
	base_pos += delta_pos;

	for(int i=0; i<children.size(); ++i) {
		if(!children[i]->is_accepted) {
			delete children[i];
			children.erase(children.begin()+i);
			if(sel>=i) --sel; // TODO: this will behave weirdly if an empty prediction was selected
			--i;
		} else {
			children[i]->reroot(delta_depth, delta_pos);
		}
	}
}

void TTE::clear_children()
{	
	for(TTE *a : children) {
		delete a;
	}
	children.clear();
}


TTE::~TTE()
{
	//printf("del %lX: '%s' (%d) at %d (+%d)\n", this, str.c_str(), tok, depth, base_pos);
	clear_children();
	// invalidate the owning buffer's LLM state if it was representing this TTE
	if(buffer->ctx_state == this) {
		buffer->ctx_state = NULL;
	}
}

TTE::TTE(LLMBuffer *b)
{
	buffer = b;
}
