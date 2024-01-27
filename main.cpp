#include "mainwindow.h"
#include <gtkmm/application.h>

#include "common.h"

CMainWindow *mainwindow;

#ifdef _WIN32
#include <windows.h>

int WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
#else
int main (int argc, char *argv[])
#endif
{
#ifdef _WIN32
	if (g_getenv("NK_WIN32_DEBUG") && AllocConsole()) {
		freopen("CONOUT$", "w", stdout);
		freopen("CONOUT$", "w", stderr);
	}
	Glib::RefPtr<Gtk::Application> app = Gtk::Application::create("com.github.blackhole89.autopen");
#else	
	Glib::RefPtr<Gtk::Application> app = Gtk::Application::create(argc, argv, "com.github.blackhole89.autopen");
#endif
	
	app->signal_activate().connect( [app,&mainwindow]() {
			mainwindow=new CMainWindow(app);
			app->add_window(*mainwindow);
		} );
		
	app->set_accel_for_action("autopen.down","<Alt>Down");
	app->set_accel_for_action("autopen.up","<Alt>Up");
	app->set_accel_for_action("autopen.right","<Alt>Right");
	app->set_accel_for_action("autopen.left","<Alt>Left");

	auto ret = app->run();


	if(mainwindow != nullptr){
		delete mainwindow;
	}

	return ret;
}
