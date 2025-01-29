#ifndef TOKENTREE_H
#define TOKENTREE_H

#include <list>
#include <thread>
#include <mutex>
#include <functional>
#include "common.h"

struct LLMBuffer;

struct TTE {
	bool is_accepted;
	
	int base_pos;
	int depth;
	
	std::shared_ptr<uint8_t[]> ctx_snapshot;
	
	llama_token tok;
	std::string str;
	int str_size;
	void set_tok(llama_token t); // compute str and size
	float logit;
	float max_logit;
	bool has_logit;

	std::vector<TTE* > children;
	TTE *parent;
	int sel;
	
	LLMBuffer *buffer;
	TTE(LLMBuffer *b);
	
	void clear_children();
	~TTE();
};

enum workload_type { WL_SCORE=0, WL_PREDICT, WL_BRANCH };

struct TTWorkload {
	workload_type wl_type;
	int base_pos;
	int tail_pos;
	int depth;
	TTE *target;
	int gen_extra;
};

struct LLMBuffer {
	TTE root;
	
	llama_model *model;
	llama_context_params ctx_params;
	llama_context *ctx;
	const llama_vocab *vocab;
	int n_vocab;
	
	std::function<void(int,int)> notify_invalidate;
	std::function<void(int,int,float)> notify_new_logit;
	std::function<void(void)> notify_new_predictions;
	std::function<void(int,std::string)> notify_change_tail;

	/* config */
	int snapshot_freq = 10;
	int predict_main = 6;
	int predict_alt = 4;

	/* model params */
	std::string model_fn, model_arch, model_size;

	/* work queue */
	std::list<TTWorkload> wq;
	bool wq_head_invalid;
	TTE *ctx_state; // token tree entry that the context has last generated / will have after work_batch is executed
	std::thread *wthread;
	void enqueueWork(workload_type, TTE *target, int gen_extra=0);
	void injectWork(workload_type, TTE *target, int gen_extra=0);
	void purgeWork(int start_depth);
	void purgePredictionWork();
	
	llama_batch work_batch;
	TTE *work_base;
	bool is_working;
	bool llm_state_changed;
	//Glib::Dispatcher work_done;
	bool work_done_flag;
	void CheckWork();
	void on_work_done();
	void try_start_working();
	std::shared_ptr<uint8_t[]> prepareBatch(TTWorkload *wl);
	void renderLogitsFromBatch(TTE* start, int n, llama_batch *b);
	
	void init();
	void load_model(const char *fn);
	
	void insert(int pos, std::string text);
	void erase(int from, int to);
	
	TTE *pos2ent(int pos);
	TTE *pos2wordent(int pos);
	
	std::string render(TTE *tt, int max_tok=99999, bool render_predictions=false);
	void rebuild(TTE *start, std::string text);
	void actualize(TTE *start);
	
	void req_alts_at_pos(int pos);
	void get_alts_at_pos(int pos, std::string &above, std::string &selected, std::string &below, int &delta);
	void alt_next(int pos);
	void alt_prev(int pos);
	int alt_commit(int pos);
	int alt_back(int pos);
	
	
	void debug_tte(TTE *pos);
	
	LLMBuffer() : root(this) {}
};


#endif