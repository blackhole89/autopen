#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <gtkmm.h>
#include "tokentree.h"

enum {
	WND_ACTION_UP,
	WND_ACTION_DOWN,
	WND_ACTION_LEFT,
	WND_ACTION_RIGHT
};

class CMainWindow : public Gtk::ApplicationWindow
{
public:
	CMainWindow(const Glib::RefPtr<Gtk::Application> &);
	virtual ~CMainWindow();

	/* header */
	Gtk::HeaderBar hbar;
	Gtk::MenuButton appbutton;
	
	Glib::RefPtr<Gio::Menu> menu = Gio::Menu::create();

	/* text view */
	Gtk::ScrolledWindow view_scroll;
	Gtk::TextView *t_view;
	Glib::RefPtr<Gtk::TextBuffer> t_buf;
	LLMBuffer llm;
	
	/* overlay */
	void rerender_predictions(int pos, std::string above, std::string selected, std::string below);
	Gtk::Label l_above, l_selected, l_below;
	
	Glib::RefPtr<Gio::SimpleActionGroup> actions;
	
	/* tags for text view */
	Glib::RefPtr<Gtk::TextTag> tag_grey;
	Glib::RefPtr<Gtk::TextTag> colortags[10];
	
	void on_text_view_allocate(Gtk::Allocation &a);
	bool suppress_signals;
	void on_insert(const Gtk::TextBuffer::iterator &iter,const Glib::ustring& str,int len);
	void on_erase(const Gtk::TextBuffer::iterator &from,const Gtk::TextBuffer::iterator &to);
	int last_position;
	void on_move_cursor();
protected:
	/* low level window for csd check */
	GdkWindow* window;

	//Signal handlers:
	void on_action(std::string name, int type, int param);

	sigc::connection idle_timer;
	bool on_idle();
};

#endif // MAINWINDOW_H
