#ifndef TOKENTREE_H
#define TOKENTREE_H

#include <list>
#include <thread>
#include <mutex>
#include <gtkmm.h>
#include "common.h"

struct TTE {
    bool is_accepted;
	
	int base_pos;
	int depth;
	
	std::shared_ptr<uint8_t[]> ctx_snapshot;
	
    llama_token tok;
	Glib::ustring str;
	float logit;
	float max_logit;
	bool has_logit;

    std::vector<TTE*> children;
	TTE *parent;
    int sel;
	
	~TTE();
};

enum workload_type { WL_SCORE, WL_PREDICT, WL_BRANCH };

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
	int n_vocab;
	
	std::function<void(int,int)> notify_invalidate;
	std::function<void(int,int,float)> notify_new_logit;
	std::function<void(void)> notify_new_predictions;
	std::function<void(int,Glib::ustring)> notify_change_tail;
	
	/* work queue */
	std::list<TTWorkload> wq;
	bool wq_head_invalid;
	int ctx_depth;
	std::thread *wthread;
	std::shared_ptr<uint8_t[]> wlToBatch(TTWorkload *wl, llama_batch *b);
	void enqueueWork(workload_type, TTE *target, int gen_extra=0);
	void injectWork(workload_type, TTE *target, int gen_extra=0);
	void purgeWork(int start_depth);
	
	llama_batch work_batch;
	bool is_working;
	Glib::Dispatcher work_done;
	void on_work_done();
	void try_start_working();
	
	void init();
	
	void insert(int pos, Glib::ustring text);
	void erase(int from, int to);
	
	TTE *pos2ent(int pos);
	TTE *pos2wordent(int pos);
	
	Glib::ustring render(TTE *tt, int max_tok=INT_MAX, bool render_predictions=false);
	void rebuild(TTE *start, Glib::ustring text);
	void actualize(TTE *start);
	
	void req_alts_at_pos(int pos);
	void get_alts_at_pos(int pos, Glib::ustring &above, Glib::ustring &selected, Glib::ustring &below, int &delta);
	void alt_next(int pos);
	void alt_prev(int pos);
	int alt_commit(int pos);
	int alt_back(int pos);
	
	
	void debug_tte(TTE *pos);
};


#endif