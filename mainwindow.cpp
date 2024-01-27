#include "mainwindow.h"
#include "tokentree.h"
#include <clocale>
#include <iostream>
#include <fontconfig/fontconfig.h>
#include <locale.h>

#include <stdlib.h>

#ifdef _WIN32
#include <windows.h>
#endif

#ifdef GDK_WINDOWING_X11
#include <gdk/gdkx.h>
#endif
#ifdef GDK_WINDOWING_WAYLAND
#include <gdk/gdkwayland.h>
#endif

#include "common.h"

CMainWindow::CMainWindow(const Glib::RefPtr<Gtk::Application>& app) : Gtk::ApplicationWindow(app)
{
	// set up window
	set_border_width(0);
	set_default_size(900,600);

	/* load stylesheet */
	Glib::RefPtr<Gtk::CssProvider> sview_css = Gtk::CssProvider::create();
	sview_css->load_from_path("data/stylesheet.css");

	get_style_context()->add_class("autopen");
	override_background_color(Gdk::RGBA("rgba(0,0,0,0)"));
	if( GdkVisual *vrgba = gdk_screen_get_rgba_visual( get_screen()->gobj() ) ) {
		gtk_widget_set_visual(GTK_WIDGET(gobj()), vrgba );
	}
		
	/* set up menu */
	menu->append("_Export Page", "win.export");
	menu->append("_Preferences", "win.pref");
	menu->append("_About", "win.about");

	/* set up header bar */	
	hbar.set_show_close_button(true);
	hbar.set_title("Autopen");
	appbutton.set_image_from_icon_name("accessories-text-editor", Gtk::ICON_SIZE_BUTTON, true);
	appbutton.set_use_popover(true);
	appbutton.set_menu_model(menu);

	hbar.pack_start(appbutton);
	set_titlebar(hbar);

	idle_timer = Glib::signal_timeout().connect(sigc::mem_fun(this,&CMainWindow::on_idle),5000,Glib::PRIORITY_LOW);

	/* text view */
	t_buf = Gtk::TextBuffer::create();
	t_view = Gtk::make_managed<Gtk::TextView>(t_buf);
	t_view->set_name("mainView");
	t_view->get_style_context()->add_provider(sview_css,GTK_STYLE_PROVIDER_PRIORITY_APPLICATION);
	t_view->property_left_margin().set_value(32);
	t_view->property_right_margin().set_value(64);
	t_view->set_wrap_mode( Gtk::WRAP_WORD_CHAR );
	t_view->property_pixels_above_lines().set_value(20);
	t_view->property_pixels_inside_wrap().set_value(20);
	view_scroll.add(*t_view);
	add( view_scroll );
	
	/* overlay */
	l_above.set_name("predAbove");
	l_above.get_style_context()->add_provider(sview_css,GTK_STYLE_PROVIDER_PRIORITY_APPLICATION);
	t_view->add_child_in_window(l_above, Gtk::TEXT_WINDOW_WIDGET, 0, 0);
	l_selected.set_name("predSelected");
	l_selected.get_style_context()->add_provider(sview_css,GTK_STYLE_PROVIDER_PRIORITY_APPLICATION);
	t_view->add_child_in_window(l_selected, Gtk::TEXT_WINDOW_WIDGET, 0, 0);
	l_below.set_name("predBelow");
	l_below.get_style_context()->add_provider(sview_css,GTK_STYLE_PROVIDER_PRIORITY_APPLICATION);
	t_view->add_child_in_window(l_below, Gtk::TEXT_WINDOW_WIDGET, 0, 0);

	/* buffer editing signals */
	suppress_signals=false;
	t_buf->signal_insert().connect(sigc::mem_fun(this,&CMainWindow::on_insert),true);
	t_buf->signal_erase().connect(sigc::mem_fun(this,&CMainWindow::on_erase),false);
	t_buf->property_cursor_position().signal_changed().connect(sigc::mem_fun(this,&CMainWindow::on_move_cursor));
	
	tag_grey = t_buf->create_tag();
	Gdk::RGBA grey;
	grey.set_rgba(0.6,0.6,0.6,
					 1.0);
	tag_grey->property_foreground_rgba().set_value(grey);
	
	for(int i=0;i<10;++i) {
		colortags[i] = t_buf->create_tag();
		
		Gdk::RGBA red;
		red.set_rgba(1.0,0.0,0.0,
					 0.1*i);
		colortags[i]->property_background_rgba().set_value(red);
	}
	
	actions = Gio::SimpleActionGroup::create();
	insert_action_group("autopen",actions);
	#define ACTION(name,param1,param2) actions->add_action(name, sigc::bind( sigc::mem_fun(this,&CMainWindow::on_action), std::string(name), param1, param2 ) )
	ACTION("up",WND_ACTION_UP,0);
	ACTION("down",WND_ACTION_DOWN,0);
	ACTION("left",WND_ACTION_LEFT,0);
	ACTION("right",WND_ACTION_RIGHT,0);
	
	/* LLM state */
	llm.init();
	llm.notify_invalidate = [this](int from, int to) {
		printf("grey %d..%d\n",from,to);
		auto istart = t_buf->get_iter_at_offset(from);
		auto iend = t_buf->get_iter_at_offset(to);
		t_buf->remove_all_tags(istart, iend);
		t_buf->apply_tag(tag_grey, istart, iend);
	};
	llm.notify_new_logit = [this](int from, int to, float logit) {
		auto istart = t_buf->get_iter_at_offset(from);
		auto iend = t_buf->get_iter_at_offset(to);
		t_buf->remove_all_tags(istart, iend);
		int index = CLAMP(-(int)((logit-1.599)/1.6), 0, 9);
		t_buf->apply_tag(colortags[index], istart, iend);
	};
	llm.notify_new_predictions = [this]() {
		Glib::ustring above,selected,below;
		int delta;
		
		llm.get_alts_at_pos(last_position, above, selected, below, delta);
		//printf("new alts: '%s', '%s', '%s' at +%d\n", above.c_str(), selected.c_str(), below.c_str(), delta);
		rerender_predictions(last_position+delta, above, selected, below);
	};
	llm.notify_change_tail = [this](int from, std::string newtext) {
		suppress_signals=true;
		t_buf->erase(t_buf->get_iter_at_offset(from), t_buf->end());
		t_buf->insert_interactive(t_buf->get_iter_at_offset(from), newtext);
		suppress_signals=false;
	};

	show_all();
}

void CMainWindow::on_action(std::string name,int type, int param)
{
	printf("action: %s\n",name.c_str());
	
	Glib::ustring above,selected,below;
	int delta, newpos;
	
	switch(type) {
	case WND_ACTION_DOWN:
		newpos = last_position;
		llm.alt_next(last_position);
		llm.get_alts_at_pos(last_position, above, selected, below, delta);
		rerender_predictions(last_position+delta, above, selected, below);
		break;
	case WND_ACTION_UP:
		newpos = last_position;
		llm.alt_prev(last_position);
		llm.get_alts_at_pos(last_position, above, selected, below, delta);
		rerender_predictions(last_position+delta, above, selected, below);
		break;
	case WND_ACTION_RIGHT:
		newpos = llm.alt_commit(last_position);
		break;
	case WND_ACTION_LEFT:
		newpos = llm.alt_back(last_position);

		break;
	}
	t_buf->place_cursor(t_buf->get_iter_at_offset(newpos));
}

void CMainWindow::on_insert(const Gtk::TextBuffer::iterator &iter,const Glib::ustring& str,int len)
{
	if(suppress_signals) return;
	
	llm.insert(iter.get_offset()-str.size(), str.c_str()); // -str.size() necessary as we run the signal after insertion (,true)
	llm.req_alts_at_pos(iter.get_offset());
}

void CMainWindow::on_erase(const Gtk::TextBuffer::iterator &from,const Gtk::TextBuffer::iterator &to)
{
	if(suppress_signals) return;
	
	llm.erase(from.get_offset(), to.get_offset());
}

void CMainWindow::on_move_cursor()
{
	Gtk::TextIter i = t_buf->get_iter_at_mark(t_buf->get_insert());
	last_position = i.get_offset();
	llm.req_alts_at_pos(last_position);
	
	Glib::ustring above,selected,below;
	int delta;
	
	llm.get_alts_at_pos(last_position, above, selected, below, delta);
	//printf("alts: '%s', '%s', '%s' at +%d\n", above.c_str(), selected.c_str(), below.c_str(), delta);
	rerender_predictions(last_position+delta, above, selected, below);
}

void CMainWindow::rerender_predictions(int pos, std::string above, std::string selected, std::string below)
{
	/*overlay_ctx->save();
	overlay_ctx->set_source_rgba(0,0,0,0);
	overlay_ctx->rectangle(0,0,t_view.get_width(),t_view.get_height());
	overlay_ctx->set_operator(Cairo::OPERATOR_SOURCE);
	overlay_ctx->fill();
	overlay_ctx->restore(); */

	Gdk::Rectangle r;
	Gtk::TextIter i = t_buf->get_iter_at_offset(pos);
	t_view->get_iter_location(i,r);
	
	above.erase( std::remove( above.begin(), above.end(), '\n' ), above.end() );
	selected.erase( std::remove( selected.begin(), selected.end(), '\n' ), selected.end() );
	below.erase( std::remove( below.begin(), below.end(), '\n' ), below.end() );
	
	int x,y;
	t_view->buffer_to_window_coords(Gtk::TEXT_WINDOW_WIDGET,r.get_x(),r.get_y(),x,y);
	
	l_above.set_text(above);
	l_above.get_window()->set_pass_through(true);
	l_selected.set_text(selected);
	l_selected.get_window()->set_pass_through(true);
	l_below.set_text(below);
	l_below.get_window()->set_pass_through(true);
	t_view->move_child(l_above, x, y-18);
	t_view->move_child(l_selected, x, y);
	t_view->move_child(l_below, x, y+18);
}

bool CMainWindow::on_idle()
{
	return true;
}

CMainWindow::~CMainWindow()
{
	
}

