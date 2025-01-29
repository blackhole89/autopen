#pragma once

#include "imgui.h"
#include "imgui_internal.h"
#include "tokentree.h"

#undef IMSTB_TEXTEDIT_STRING
#undef IMSTB_TEXTEDIT_CHARTYPE
#define IMSTB_TEXTEDIT_STRING             ImGuiInputTextState
#define IMSTB_TEXTEDIT_CHARTYPE           char
#define IMSTB_TEXTEDIT_GETWIDTH_NEWLINE   (-1.0f)
#define IMSTB_TEXTEDIT_UNDOSTATECOUNT     99
#define IMSTB_TEXTEDIT_UNDOCHARCOUNT      999
namespace LLMStb { struct STB_TexteditState; }
typedef LLMStb::STB_TexteditState LLMStbTexteditState;

// Internal state of the currently focused/edited text input box
// For a given item ID, access with ImGui::GetInputTextState()
struct LLMTextState
{
	LLMBuffer llm;
	
	bool invalidate_predictions;
	TTE *current_tok, *last_tok;
	std::string above, selected, below;
	
    ImGuiContext*           Ctx;                    // parent UI context (needs to be set explicitly by parent).
    LLMStbTexteditState*    Stb;                    // State for stb_textedit.h
    ImGuiInputTextFlags     Flags;                  // copy of InputText() flags. may be used to check if e.g. ImGuiInputTextFlags_Password is set.
    ImGuiID                 ID;                     // widget id owning the text state
    int                     TextLen;                // UTF-8 length of the string in TextA (in bytes)
    const char*             TextSrc;                // == TextA.Data unless read-only, in which case == buf passed to InputText(). Field only set and valid _inside_ the call InputText() call.
    ImVector<char>          TextA;                  // main UTF8 buffer. TextA.Size is a buffer size! Should always be >= buf_size passed by user (and of course >= CurLenA + 1).
    ImVector<char>          TextToRevertTo;         // value to revert to when pressing Escape = backup of end-user buffer at the time of focus (in UTF-8, unaltered)
    ImVector<char>          CallbackTextBackup;     // temporary storage for callback to support automatic reconcile of undo-stack
    int                     BufCapacity;            // end-user buffer capacity (include zero terminator)
    ImVec2                  Scroll;                 // horizontal offset (managed manually) + vertical scrolling (pulled from child window's own Scroll.y)
    float                   CursorAnim;             // timer for cursor blink, reset on every user action so the cursor reappears immediately
    bool                    CursorFollow;           // set when we want scrolling to follow the current cursor position (not always!)
    bool                    SelectedAllMouseLock;   // after a double-click to select all, we ignore further mouse drags to update selection
    bool                    Edited;                 // edited this frame
    bool                    WantReloadUserBuf;      // force a reload of user buf so it may be modified externally. may be automatic in future version.
    int                     ReloadSelectionStart;
    int                     ReloadSelectionEnd;

    LLMTextState();
    ~LLMTextState();
    void        ClearText()                 { TextLen = 0; TextA[0] = 0; CursorClamp(); }
    void        ClearFreeMemory()           { TextA.clear(); TextToRevertTo.clear(); }
    void        OnKeyPressed(int key);      // Cannot be inline because we call in code in stb_textedit.h implementation
    void        OnCharPressed(unsigned int c);

    // Cursor & Selection
    void        CursorAnimReset();
    void        CursorClamp();
    bool        HasSelection() const;
    void        ClearSelection();
    int         GetCursorPos() const;
    int         GetSelectionStart() const;
    int         GetSelectionEnd() const;
    void        SelectAll();

    // Reload user buf (WIP #2890)
    // If you modify underlying user-passed const char* while active you need to call this (InputText V2 may lift this)
    //   strcpy(my_buf, "hello");
    //   if (ImGuiInputTextState* state = ImGui::GetInputTextState(id)) // id may be ImGui::GetItemID() is last item
    //       state->ReloadUserBufAndSelectAll();
    void        ReloadUserBufAndSelectAll();
    void        ReloadUserBufAndKeepSelection();
    void        ReloadUserBufAndMoveToEnd();
};

struct CEditor {
	LLMTextState llmst;
	
	ImFont *font_editor, *font_ui;
	
	bool p_wqueue = true, p_settings = false;
	
	char *buf;
	
	void Init();
	void Render();
    void SettingsWindow();
	bool EditorWidget(const char* label, const char* hint, char* buf, int buf_size, const ImVec2& size_arg, ImGuiInputTextFlags flags);
};