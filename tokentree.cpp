#include "tokentree.h"
#include <set>

void LLMBuffer::init()
{
	/* init callbacks */
	notify_invalidate = [](int,int) {};
	notify_new_logit = [](int,int,float) {};
	
	/* init llama.cpp */
	gpt_params params;
	llama_backend_init(params.numa);
	
	llama_model_params model_params = llama_model_default_params();

	// model_params.n_gpu_layers = 99; // offload all layers to the GPU
	//model = llama_load_model_from_file("llama.cpp/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", model_params);
	model = llama_load_model_from_file("llama.cpp/openhermes-2-mistral-7b.Q4_K_M.gguf", model_params);

	if (model == NULL) {
		fprintf(stderr , "%s: error: unable to load model\n" , __func__);
		exit(1);
	}
	
	// initialize the context
	ctx_params = llama_context_default_params();

	ctx_params.seed  = 1234;
	ctx_params.n_ctx = 2048;
	ctx_params.n_threads = params.n_threads;
	ctx_params.n_threads_batch = params.n_threads_batch == -1 ? params.n_threads : params.n_threads_batch;

	ctx = llama_new_context_with_model(model, ctx_params);

	if (ctx == NULL) {
		fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
		exit(1);
	}
	
	n_vocab = llama_n_vocab(model);
	
	/* init root token */
	root.base_pos=0;
	root.depth=0;
	root.is_accepted=true;
	root.set_tok(1);
	root.str="";
	root.parent=NULL;
	root.sel=0;
	root.ctx_snapshot = std::shared_ptr<uint8_t[]>(new uint8_t[llama_get_state_size(ctx)]);
	llama_copy_state_data(ctx,root.ctx_snapshot.get());
	ctx_state=NULL;
	
	// start work thread
	work_batch = llama_batch_init(512, 0, 1);
	work_base = &root;
	is_working = false;
	wq_head_invalid = false;
	work_done.connect(sigc::mem_fun(this,&LLMBuffer::on_work_done));
	
	/*wthread = new std::thread(
	  [this]
	  {
		workThread();
	  });*/
	
}

void LLMBuffer::insert(int pos, Glib::ustring text)
{
	TTE *start = pos2wordent(pos);
	Glib::ustring tail = render(start);
	printf("pre-ins: '%s', '%s', %d\n", tail.c_str(), text.c_str(), pos - start->base_pos);
	tail.insert(pos - start->base_pos, text);
	printf("post-ins: '%s'\n", tail.c_str());
	
	rebuild(start, tail);
}

void LLMBuffer::erase(int from, int to)
{
	TTE *start = pos2wordent(from);
	Glib::ustring tail = render(start);
	tail.erase(from - start->base_pos, to - from);
	
	rebuild(start, tail);
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
	
	while(cur->parent && cur->str.find(' ')==Glib::ustring::npos) {
		cur=cur->parent;
pos2wordent_end:;
//		printf("back: %lX, '%s'\n", cur->parent, cur->str.c_str());
	}
	
	return cur;
}

/* convert live path from token tree entry to string */
Glib::ustring LLMBuffer::render(TTE *tt, int max_tok, bool render_predictions)
{
	Glib::ustring ret;
	while(tt && max_tok && (render_predictions || tt->is_accepted)) {
		ret += tt->str;
		tt = ((tt->children.size()>0)&&(tt->sel>=0))?(TTE*)tt->children[tt->sel]:NULL;
		--max_tok;
	}
	return ret;
}

void LLMBuffer::rebuild(TTE *start, Glib::ustring text)
{
	std::vector<llama_token> tokens_list = llama_tokenize(ctx, text, start->tok==1,  true);
	
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
	start->clear_children(); 
	for(size_t i=0;i<tokens_list.size();++i) {
		start ->has_logit = false;
		start->set_tok(tokens_list[i]); 
		// NOTE: Need patched llama.cpp to not add extra space.
		//if(start->base_pos==0 && start->str[0]==' ') start->str.erase(0,1);
		
		printf("%d '%s' . ",start->tok, start->str.c_str());
		
		if((i+1)<tokens_list.size()) {
			start->children.push_back(new TTE(this));
			start->sel=0;
			TTE *next = start->children[0];
			next->base_pos = start->base_pos + start->str_size;
			next->depth = start->depth + 1;
			next->parent = start;
			next->is_accepted = true;
			next->sel = 0;
			start=next;
		}
	}
	printf("\n");
	
	enqueueWork(WL_SCORE, rebuild_root);
}

void LLMBuffer::actualize(TTE *start)
{
	Glib::ustring txt = render(start);
	/* leap over token to get valid UTF-8 */
	if(!txt.validate()) {
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
			
		while(extend() && !txt.validate());
		if(!txt.validate()) {
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
		//printf("purge '%s'\n", wq.front().target->str.c_str());
	}
	for(auto i = ++wq.begin(), j=i; i!=wq.end(); ) {
		++j;
		
		if(i->depth >= start_depth) {
			//printf("purge '%s'\n", i->target->str.c_str());
			wq.erase(i);
		}
		
		i=j;
	}
}

void LLMBuffer::purgePredictionWork()
{
	for(auto i = ++wq.begin(), j=i; i!=wq.end(); ) {
		++j;
		
		if(i->wl_type == WL_BRANCH || i->wl_type == WL_PREDICT) {
			wq.erase(i);
		}
		
		i=j;
	}
}

void LLMBuffer::on_work_done()
{
	is_working=false;
	
	if(!wq_head_invalid) {
		std::shared_ptr<uint8_t[]> snap;
		if(llm_state_changed && ((work_base->depth%10)+work_batch.n_tokens)>=10)  {
			printf("snap, as work base is at %d and processed %d extra tokens\n", work_base->depth, work_batch.n_tokens);
			snap = std::shared_ptr<uint8_t[]>(new uint8_t[llama_get_state_size(ctx)]);
			llama_copy_state_data(ctx,snap.get());
		}
		
		switch(wq.front().wl_type) {
		case WL_SCORE: {
			float *logits = llama_get_logits_ith(ctx, work_batch.n_tokens - 1);
			float max_logit = *std::max_element(logits, logits+n_vocab);
			
			TTE *t = wq.front().target;
			int gen_extra = wq.front().gen_extra;
			
			wq.pop_front();
			wq_head_invalid=false;
			
			/* TODO make sure it's okay to skip if even just the selected child has a logit already */
			if(!t->children.size() || t->children[t->sel]->has_logit) {
				if(t->children.size()) injectWork(WL_SCORE, t->children[t->sel], gen_extra);
				break;
			}
			
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
					injectWork(WL_PREDICT, t->children[t->sel], gen_extra-1);
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
				
				exclude.insert(i_max);
			}
			
			notify_new_predictions();
			
			if(gen_extra>0) {
				injectWork(WL_PREDICT, t->children[t->sel], gen_extra-1);
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
		prepareBatch(&wq.front());

		is_working = true;
		wthread = new std::thread(
		  [this]
		  {
			llama_decode(ctx, work_batch);
			llm_state_changed = true;
			work_done.emit();
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

void LLMBuffer::prepareBatch(TTWorkload *wl)
{
	llama_batch_clear(work_batch);
	if(ctx_state && ctx_state == wl->target->parent) {
		llama_batch_add(work_batch, wl->target->tok, wl->target->depth, { 0 }, false);
		work_batch.logits[0]=true;
		work_base = wl->target;
		ctx_state = wl->target;
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
			llama_batch_add(work_batch, toks[i], d++, { 0 }, false);
			work_batch.logits[toks.size()-1-i] = need_logits[i];
		}
		
		// reset context to snapshot
		llama_free(ctx);
		ctx = llama_new_context_with_model(model, ctx_params);
		//llama_reinit_kv_cache(ctx, ctx_params);
		int n_copied = llama_set_state_data(ctx,pos->ctx_snapshot.get());
		
		// debug output
		std::string txt;
		for(int i = toks.size()-1; i>=0; --i) {
			txt+=llama_token_to_piece(ctx,toks[i]);
		}
		printf("reset to '%s' (%d) at %d (+%d), catchup '%s'\n", pos->str.c_str(), pos->tok, pos->depth, pos->base_pos, txt.c_str());
		
		work_base = pos;
		ctx_state = wl->target;
	}
}

void LLMBuffer::req_alts_at_pos(int pos)
{
	TTE *cur = pos2ent(pos);
	printf("req alts from '%s' (%d) at %d (+%d)\n", cur->str.c_str(), cur->tok, cur->depth, cur->base_pos);
	purgePredictionWork(); // get rid of old prediction tasks
	injectWork(WL_BRANCH, cur, 4);
	try_start_working();
}

void LLMBuffer::get_alts_at_pos(int pos, Glib::ustring &above, Glib::ustring &selected, Glib::ustring &below, int &delta)
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
		
	if(cur->children.size()) actualize(cur->children[cur->sel]);
		
	injectWork(WL_BRANCH, cur, 4);
	try_start_working();
}

void LLMBuffer::alt_prev(int pos)
{
	TTE *cur = pos2ent(pos);
	if(cur->sel>0)
		--cur->sel;
	
	if(cur->children.size()) actualize(cur->children[cur->sel]);
	
	injectWork(WL_BRANCH, cur, 4);
	try_start_working();
}

int LLMBuffer::alt_commit(int pos)
{
	TTE *cur = pos2ent(pos);
	if(cur->children.size()) {
		cur->children[cur->sel]->is_accepted = true;
		actualize(cur->children[cur->sel]);
		return cur->children[cur->sel]->base_pos + cur->children[cur->sel]->str_size;
	} else return cur->base_pos + cur->str_size;
}

int LLMBuffer::alt_back(int pos)
{
	TTE *cur = pos2ent(pos);
	
	if(cur->base_pos == pos) {
		if(cur->parent) return cur->parent->base_pos;
		else return cur->base_pos;
	} else return cur->base_pos;
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
	tok = t;
	str = llama_token_to_piece(buffer->ctx, t);
	if(str.validate()) str_size = str.size();
	else {
		if((((unsigned char)str.c_str()[0])&0xC0) != 0x80) str_size=1;
		else str_size=0;
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