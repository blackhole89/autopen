#ifndef IMGUI_DEFINE_MATH_OPERATORS
#define IMGUI_DEFINE_MATH_OPERATORS
#endif

#include "imgui.h"
#include "imgui_internal.h"

#include "editor.h"

#include <algorithm>

#include "imfilebrowser/imfilebrowser.h"

namespace LLMStb {
static bool ReplaceTail(LLMTextState* obj, int pos, const char* new_text, int new_text_len);
}

void CEditor::Init()
{
	llmst.llm.init();
	llmst.Ctx = ImGui::GetCurrentContext();
	llmst.llm.notify_new_predictions = [this]() { llmst.invalidate_predictions=true; };
	llmst.llm.notify_change_tail = [this](int n, std::string u) { LLMStb::ReplaceTail(&llmst, n, u.c_str(), u.length() ); };
	
	buf = new char[40960];
	buf[0]=0;
}

void CEditor::SettingsWindow()
{
    if(p_settings) {
        ImGui::SetNextWindowSize(ImVec2(300, 300), ImGuiCond_FirstUseEver);
        if(ImGui::Begin("Settings##sw", &p_settings)) {
            ImGui::PushItemWidth(100);
            ImGui::InputInt("Snapshot interval", &llmst.llm.snapshot_freq);
            ImGui::SetItemTooltip("LLM state will be stored at this interval. Lower values make predictions faster, but use more RAM.");
            
            ImGui::InputInt("Main prediction depth", &llmst.llm.predict_main);
            ImGui::SetItemTooltip("How many tokens to predict for the currently selected branch");

            ImGui::InputInt("Side prediction depth", &llmst.llm.predict_alt);
            ImGui::SetItemTooltip("How many tokens to predict for alternative branches");

            ImGui::PopItemWidth();

            ImGui::Separator();

            static ImGui::FileBrowser fileDialog;

            fileDialog.SetTitle("Select GGUF model");
            fileDialog.SetTypeFilters({".gguf"});

            ImGui::Text("Loaded model: "); ImGui::SameLine();
            if(ImGui::Button(llmst.llm.model_fn.c_str())) 
                fileDialog.Open();

            fileDialog.Display();
            
            if(fileDialog.HasSelected()) {
                llmst.llm.load_model(fileDialog.GetSelected().string().c_str());
                fileDialog.ClearSelected();
            }

            ImGui::Text("Type: %s %s",llmst.llm.model_arch.c_str(), llmst.llm.model_size.c_str());

            ImGui::Text("Params: %lld   Layers: %d   Heads: %d", llama_model_n_params(llmst.llm.model), llama_model_n_layer(llmst.llm.model), llama_model_n_head(llmst.llm.model));

            bool open = ImGui::CollapsingHeader("Model metadata");
            if(open) {
                for(int i=0; i<llama_model_meta_count(llmst.llm.model); i++) {
                    char k_buf[256], v_buf[256];
                    llama_model_meta_key_by_index(llmst.llm.model, i, k_buf, 256);
                    llama_model_meta_val_str_by_index(llmst.llm.model, i, v_buf, 256);
                
                    ImGui::Text("%s = %s", k_buf, v_buf);
                }
            }
            
        }
        ImGui::End();
    }
}

void CEditor::Render()
{
	llmst.llm.CheckWork();

	ImGui::PushFont(font_ui);
	
    if (ImGui::BeginMainMenuBar())
    {
        if (ImGui::BeginMenu("File"))
        {
            ImGui::MenuItem("Load buffer", NULL, false, false);
            ImGui::MenuItem("Save buffer", NULL, false, false);
            ImGui::Separator();
            if(ImGui::MenuItem("Settings", NULL, false, true))
                p_settings = true;
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Windows"))
        {
            ImGui::MenuItem("Work queue", NULL, &p_wqueue);
			ImGui::EndMenu();
		}
		ImGui::EndMainMenuBar();
	}
	
	ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoBringToFrontOnFocus;

    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->WorkPos);
    ImGui::SetNextWindowSize(viewport->WorkSize);
	
	bool p_open = true;
	ImGui::Begin("Editor", &p_open, flags);
	
	ImGui::PushFont(font_editor);
	EditorWidget("##source", "", buf, 40960, ImVec2(-FLT_MIN, -20.0), ImGuiInputTextFlags_Multiline | ImGuiInputTextFlags_NoUndoRedo);
	ImGui::PopFont();
	
	if(llmst.current_tok) {
		ImGui::Text("DEPTH: %3d (+%3d) -- CHILDREN: %d/%d -- LOG.L: %2.3f -- TOP: %2.3f -- TOK: %d '%s'",
			llmst.current_tok->depth, llmst.current_tok->base_pos, llmst.current_tok->sel, llmst.current_tok->children.size(), llmst.current_tok->logit, llmst.current_tok->max_logit, llmst.current_tok->tok, llmst.current_tok->str.c_str());
	}
	
	ImGui::End();
	
	//ImGui::ShowDemoWindow();
	if(p_wqueue) {
        ImGui::SetNextWindowSize(ImVec2(300, 300), ImGuiCond_FirstUseEver);
        if(ImGui::Begin("Work queue", &p_wqueue)) {
		    if (ImGui::BeginListBox("##wq", ImVec2(-FLT_MIN, -FLT_MIN)))
		    {
			    const char *wl_typenames[3] = { "SCORE  ", "PREDICT", "BRANCH " };
			    for(auto i = llmst.llm.wq_head_invalid?++llmst.llm.wq.begin():llmst.llm.wq.begin(); i!=llmst.llm.wq.end(); ++i) {
				    ImGui::Text("%s %16lX %d (+%d) '%s'..", wl_typenames[i->wl_type], i->target, i->depth, i->base_pos, i->target->str.c_str());
			    }
			    ImGui::EndListBox();
		    }
        }
		ImGui::End();
	}

    SettingsWindow();
	
	ImGui::PopFont();
}

/* editor widget */

using namespace ImGui;

static bool     InputTextFilterCharacter(ImGuiContext* ctx, unsigned int* p_char, ImGuiInputTextFlags flags, ImGuiInputTextCallback callback, void* user_data, bool input_source_is_clipboard = false);
static int      InputTextCalcTextLenAndLineCount(const char* text_begin, const char** out_text_end);
static ImVec2   InputTextCalcTextSize(ImGuiContext* ctx, const char* text_begin, const char* text_end, const char** remaining = NULL, ImVec2* out_offset = NULL, bool stop_on_new_line = false);

namespace LLMStb
{
#undef INCLUDE_IMSTB_TEXTEDIT_H
#include "imstb_textedit.h"
}

// This is only used in the path where the multiline widget is inactivate.
static int InputTextCalcTextLenAndLineCount(const char* text_begin, const char** out_text_end)
{
    int line_count = 0;
    const char* s = text_begin;
    while (true)
    {
        const char* s_eol = strchr(s, '\n');
        line_count++;
        if (s_eol == NULL)
        {
            s = s + strlen(s);
            break;
        }
        s = s_eol + 1;
    }
    *out_text_end = s;
    return line_count;
}

// FIXME: Ideally we'd share code with ImFont::CalcTextSizeA()
static ImVec2 InputTextCalcTextSize(ImGuiContext* ctx, const char* text_begin, const char* text_end, const char** remaining, ImVec2* out_offset, bool stop_on_new_line)
{
    ImGuiContext& g = *ctx;
    ImFont* font = g.Font;
    const float line_height = g.FontSize;
    const float scale = line_height / font->FontSize;

    ImVec2 text_size = ImVec2(0, 0);
    float line_width = 0.0f;

    const char* s = text_begin;
    while (s < text_end)
    {
        unsigned int c = (unsigned int)*s;
        if (c < 0x80)
            s += 1;
        else
            s += ImTextCharFromUtf8(&c, s, text_end);

        if (c == '\n')
        {
            text_size.x = ImMax(text_size.x, line_width);
            text_size.y += line_height;
            line_width = 0.0f;
            if (stop_on_new_line)
                break;
            continue;
        }
        if (c == '\r')
            continue;

        const float char_width = ((int)c < font->IndexAdvanceX.Size ? font->IndexAdvanceX.Data[c] : font->FallbackAdvanceX) * scale;
        line_width += char_width;
    }

    if (text_size.x < line_width)
        text_size.x = line_width;

    if (out_offset)
        *out_offset = ImVec2(line_width, text_size.y + line_height);  // offset allow for the possibility of sitting after a trailing \n

    if (line_width > 0 || text_size.y == 0.0f)                        // whereas size.y will ignore the trailing \n
        text_size.y += line_height;

    if (remaining)
        *remaining = s;

    return text_size;
}


namespace LLMStb
{
#undef IMSTB_TEXTEDIT_STRING
#define IMSTB_TEXTEDIT_STRING             LLMTextState

static int     STB_TEXTEDIT_STRINGLEN(const LLMTextState* obj)                             { return obj->TextLen; }
static char    STB_TEXTEDIT_GETCHAR(const LLMTextState* obj, int idx)                      { IM_ASSERT(idx <= obj->TextLen); return obj->TextSrc[idx]; }
static float   STB_TEXTEDIT_GETWIDTH(LLMTextState* obj, int line_start_idx, int char_idx)  { unsigned int c; ImTextCharFromUtf8(&c, obj->TextSrc + line_start_idx + char_idx, obj->TextSrc + obj->TextLen); if ((ImWchar)c == '\n') return IMSTB_TEXTEDIT_GETWIDTH_NEWLINE; ImGuiContext& g = *obj->Ctx; return g.Font->GetCharAdvance((ImWchar)c) * g.FontScale; }
static char    STB_TEXTEDIT_NEWLINE = '\n';
static void    STB_TEXTEDIT_LAYOUTROW(StbTexteditRow* r, LLMTextState* obj, int line_start_idx)
{
    const char* text = obj->TextSrc;
    const char* text_remaining = NULL;
    const ImVec2 size = InputTextCalcTextSize(obj->Ctx, text + line_start_idx, text + obj->TextLen, &text_remaining, NULL, true);
    r->x0 = 0.0f;
    r->x1 = size.x;
    r->baseline_y_delta = size.y;
    r->ymin = 0.0f;
    r->ymax = size.y;
    r->num_chars = (int)(text_remaining - (text + line_start_idx));
}

#define IMSTB_TEXTEDIT_GETNEXTCHARINDEX  IMSTB_TEXTEDIT_GETNEXTCHARINDEX_IMPL
#define IMSTB_TEXTEDIT_GETPREVCHARINDEX  IMSTB_TEXTEDIT_GETPREVCHARINDEX_IMPL

static int IMSTB_TEXTEDIT_GETNEXTCHARINDEX_IMPL(LLMTextState* obj, int idx)
{
    if (idx >= obj->TextLen)
        return obj->TextLen + 1;
    unsigned int c;
    return idx + ImTextCharFromUtf8(&c, obj->TextSrc + idx, obj->TextSrc + obj->TextLen);
}

static int IMSTB_TEXTEDIT_GETPREVCHARINDEX_IMPL(LLMTextState* obj, int idx)
{
    if (idx <= 0)
        return -1;
    const char* p = ImTextFindPreviousUtf8Codepoint(obj->TextSrc, obj->TextSrc + idx);
    return (int)(p - obj->TextSrc);
}

static bool ImCharIsSeparatorW(unsigned int c)
{
    static const unsigned int separator_list[] =
    {
        ',', 0x3001, '.', 0x3002, ';', 0xFF1B, '(', 0xFF08, ')', 0xFF09, '{', 0xFF5B, '}', 0xFF5D,
        '[', 0x300C, ']', 0x300D, '|', 0xFF5C, '!', 0xFF01, '\\', 0xFFE5, '/', 0x30FB, 0xFF0F,
        '\n', '\r',
    };
    for (unsigned int separator : separator_list)
        if (c == separator)
            return true;
    return false;
}

static int is_word_boundary_from_right(LLMTextState* obj, int idx)
{
    // When ImGuiInputTextFlags_Password is set, we don't want actions such as CTRL+Arrow to leak the fact that underlying data are blanks or separators.
    if ((obj->Flags & ImGuiInputTextFlags_Password) || idx <= 0)
        return 0;

    const char* curr_p = obj->TextSrc + idx;
    const char* prev_p = ImTextFindPreviousUtf8Codepoint(obj->TextSrc, curr_p);
    unsigned int curr_c; ImTextCharFromUtf8(&curr_c, curr_p, obj->TextSrc + obj->TextLen);
    unsigned int prev_c; ImTextCharFromUtf8(&prev_c, prev_p, obj->TextSrc + obj->TextLen);

    bool prev_white = ImCharIsBlankW(prev_c);
    bool prev_separ = ImCharIsSeparatorW(prev_c);
    bool curr_white = ImCharIsBlankW(curr_c);
    bool curr_separ = ImCharIsSeparatorW(curr_c);
    return ((prev_white || prev_separ) && !(curr_separ || curr_white)) || (curr_separ && !prev_separ);
}
static int is_word_boundary_from_left(LLMTextState* obj, int idx)
{
    if ((obj->Flags & ImGuiInputTextFlags_Password) || idx <= 0)
        return 0;

    const char* curr_p = obj->TextSrc + idx;
    const char* prev_p = ImTextFindPreviousUtf8Codepoint(obj->TextSrc, curr_p);
    unsigned int prev_c; ImTextCharFromUtf8(&prev_c, curr_p, obj->TextSrc + obj->TextLen);
    unsigned int curr_c; ImTextCharFromUtf8(&curr_c, prev_p, obj->TextSrc + obj->TextLen);

    bool prev_white = ImCharIsBlankW(prev_c);
    bool prev_separ = ImCharIsSeparatorW(prev_c);
    bool curr_white = ImCharIsBlankW(curr_c);
    bool curr_separ = ImCharIsSeparatorW(curr_c);
    return ((prev_white) && !(curr_separ || curr_white)) || (curr_separ && !prev_separ);
}
static int  STB_TEXTEDIT_MOVEWORDLEFT_IMPL(LLMTextState* obj, int idx)
{
    idx = IMSTB_TEXTEDIT_GETPREVCHARINDEX(obj, idx);
    while (idx >= 0 && !is_word_boundary_from_right(obj, idx))
        idx = IMSTB_TEXTEDIT_GETPREVCHARINDEX(obj, idx);
    return idx < 0 ? 0 : idx;
}
static int  STB_TEXTEDIT_MOVEWORDRIGHT_MAC(LLMTextState* obj, int idx)
{
    int len = obj->TextLen;
    idx = IMSTB_TEXTEDIT_GETNEXTCHARINDEX(obj, idx);
    while (idx < len && !is_word_boundary_from_left(obj, idx))
        idx = IMSTB_TEXTEDIT_GETNEXTCHARINDEX(obj, idx);
    return idx > len ? len : idx;
}
static int  STB_TEXTEDIT_MOVEWORDRIGHT_WIN(LLMTextState* obj, int idx)
{
    idx = IMSTB_TEXTEDIT_GETNEXTCHARINDEX(obj, idx);
    int len = obj->TextLen;
    while (idx < len && !is_word_boundary_from_right(obj, idx))
        idx = IMSTB_TEXTEDIT_GETNEXTCHARINDEX(obj, idx);
    return idx > len ? len : idx;
}
static int  STB_TEXTEDIT_MOVEWORDRIGHT_IMPL(LLMTextState* obj, int idx)  { ImGuiContext& g = *obj->Ctx; if (g.IO.ConfigMacOSXBehaviors) return STB_TEXTEDIT_MOVEWORDRIGHT_MAC(obj, idx); else return STB_TEXTEDIT_MOVEWORDRIGHT_WIN(obj, idx); }
#define STB_TEXTEDIT_MOVEWORDLEFT       STB_TEXTEDIT_MOVEWORDLEFT_IMPL  // They need to be #define for stb_textedit.h
#define STB_TEXTEDIT_MOVEWORDRIGHT      STB_TEXTEDIT_MOVEWORDRIGHT_IMPL

static void STB_TEXTEDIT_DELETECHARS(LLMTextState* obj, int pos, int n)
{
    // Offset remaining text (+ copy zero terminator)
    IM_ASSERT(obj->TextSrc == obj->TextA.Data);
    char* dst = obj->TextA.Data + pos;
    char* src = obj->TextA.Data + pos + n;
    memmove(dst, src, obj->TextLen - n - pos + 1);
	
	// notify LLM
	obj->llm.erase(pos, pos+n);
	obj->llm.req_alts_at_pos(pos);
	obj->last_tok = NULL; 
	
    obj->Edited = true;
    obj->TextLen -= n;
}

static bool STB_TEXTEDIT_INSERTCHARS(LLMTextState* obj, int pos, const char* new_text, int new_text_len)
{
    const bool is_resizable = (obj->Flags & ImGuiInputTextFlags_CallbackResize) != 0;
    const int text_len = obj->TextLen;
    IM_ASSERT(pos <= text_len);

    if (!is_resizable && (new_text_len + obj->TextLen + 1 > obj->BufCapacity))
        return false;

    // Grow internal buffer if needed
    IM_ASSERT(obj->TextSrc == obj->TextA.Data);
    if (new_text_len + text_len + 1 > obj->TextA.Size)
    {
        if (!is_resizable)
            return false;
        obj->TextA.resize(text_len + ImClamp(new_text_len, 32, ImMax(256, new_text_len)) + 1);
        obj->TextSrc = obj->TextA.Data;
    }

    char* text = obj->TextA.Data;
    if (pos != text_len)
        memmove(text + pos + new_text_len, text + pos, (size_t)(text_len - pos));
    memcpy(text + pos, new_text, (size_t)new_text_len);
	
	// notify LLM
	obj->llm.insert(pos, new_text);
	obj->llm.req_alts_at_pos(pos + new_text_len);

    obj->Edited = true;
    obj->TextLen += new_text_len;
    obj->TextA[obj->TextLen] = '\0';

    return true;
}

static bool ReplaceTail(LLMTextState* obj, int pos, const char* new_text, int new_text_len)
{
	const bool is_resizable = (obj->Flags & ImGuiInputTextFlags_CallbackResize) != 0;
    const int text_len = obj->TextLen;
    IM_ASSERT(pos <= text_len);

    if (!is_resizable && (new_text_len + obj->TextLen + 1 > obj->BufCapacity))
        return false;

    // Grow internal buffer if needed
    IM_ASSERT(obj->TextSrc == obj->TextA.Data);
    if (new_text_len + pos + 1 > obj->TextA.Size)
    {
        if (!is_resizable)
            return false;
        obj->TextA.resize(pos + ImMin(new_text_len, 32) + 1);
        obj->TextSrc = obj->TextA.Data;
    }

    char* text = obj->TextA.Data;
    memcpy(text + pos, new_text, (size_t)new_text_len);
	
    obj->Edited = true;
    obj->TextLen = pos + new_text_len;
    obj->TextA[obj->TextLen] = '\0';

    return true;
}

// We don't use an enum so we can build even with conflicting symbols (if another user of stb_textedit.h leak their STB_TEXTEDIT_K_* symbols)
#define STB_TEXTEDIT_K_LEFT         0x200000 // keyboard input to move cursor left
#define STB_TEXTEDIT_K_RIGHT        0x200001 // keyboard input to move cursor right
#define STB_TEXTEDIT_K_UP           0x200002 // keyboard input to move cursor up
#define STB_TEXTEDIT_K_DOWN         0x200003 // keyboard input to move cursor down
#define STB_TEXTEDIT_K_LINESTART    0x200004 // keyboard input to move cursor to start of line
#define STB_TEXTEDIT_K_LINEEND      0x200005 // keyboard input to move cursor to end of line
#define STB_TEXTEDIT_K_TEXTSTART    0x200006 // keyboard input to move cursor to start of text
#define STB_TEXTEDIT_K_TEXTEND      0x200007 // keyboard input to move cursor to end of text
#define STB_TEXTEDIT_K_DELETE       0x200008 // keyboard input to delete selection or character under cursor
#define STB_TEXTEDIT_K_BACKSPACE    0x200009 // keyboard input to delete selection or character left of cursor
#define STB_TEXTEDIT_K_UNDO         0x20000A // keyboard input to perform undo
#define STB_TEXTEDIT_K_REDO         0x20000B // keyboard input to perform redo
#define STB_TEXTEDIT_K_WORDLEFT     0x20000C // keyboard input to move cursor left one word
#define STB_TEXTEDIT_K_WORDRIGHT    0x20000D // keyboard input to move cursor right one word
#define STB_TEXTEDIT_K_PGUP         0x20000E // keyboard input to move cursor up a page
#define STB_TEXTEDIT_K_PGDOWN       0x20000F // keyboard input to move cursor down a page
#define STB_TEXTEDIT_K_SHIFT        0x400000

#define IMSTB_TEXTEDIT_IMPLEMENTATION
#define IMSTB_TEXTEDIT_memmove memmove
#include "imstb_textedit.h"

// stb_textedit internally allows for a single undo record to do addition and deletion, but somehow, calling
// the stb_textedit_paste() function creates two separate records, so we perform it manually. (FIXME: Report to nothings/stb?)
static void stb_textedit_replace(LLMTextState* str, STB_TexteditState* state, const IMSTB_TEXTEDIT_CHARTYPE* text, int text_len)
{
    stb_text_makeundo_replace(str, state, 0, str->TextLen, text_len);
    LLMStb::STB_TEXTEDIT_DELETECHARS(str, 0, str->TextLen);
    state->cursor = state->select_start = state->select_end = 0;
    if (text_len <= 0)
        return;
    if (LLMStb::STB_TEXTEDIT_INSERTCHARS(str, 0, text, text_len))
    {
        state->cursor = state->select_start = state->select_end = text_len;
        state->has_preferred_x = 0;
        return;
    }
    IM_ASSERT(0); // Failed to insert character, normally shouldn't happen because of how we currently use stb_textedit_replace()
}

} // namespace LLMStb

LLMTextState::LLMTextState()
{
    //memset(this, 0, sizeof(*this));  // -- this destroys our root node, so let's not
    Stb = IM_NEW(LLMStbTexteditState);
    memset(Stb, 0, sizeof(*Stb));
}

LLMTextState::~LLMTextState()
{
    IM_DELETE(Stb);
}

void LLMTextState::OnKeyPressed(int key)
{
    stb_textedit_key(this, Stb, key);
    CursorFollow = true;
    CursorAnimReset();
}

void LLMTextState::OnCharPressed(unsigned int c)
{
    // Convert the key to a UTF8 byte sequence.
    // The changes we had to make to stb_textedit_key made it very much UTF-8 specific which is not too great.
    char utf8[5];
    ImTextCharToUtf8(utf8, c);
    stb_textedit_text(this, Stb, utf8, (int)strlen(utf8));
    CursorFollow = true;
    CursorAnimReset();
}

// Those functions are not inlined in imgui_internal.h, allowing us to hide ImStbTexteditState from that header.
void LLMTextState::CursorAnimReset()                 { CursorAnim = -0.30f; } // After a user-input the cursor stays on for a while without blinking
void LLMTextState::CursorClamp()                     { Stb->cursor = ImMin(Stb->cursor, TextLen); Stb->select_start = ImMin(Stb->select_start, TextLen); Stb->select_end = ImMin(Stb->select_end, TextLen); }
bool LLMTextState::HasSelection() const              { return Stb->select_start != Stb->select_end; }
void LLMTextState::ClearSelection()                  { Stb->select_start = Stb->select_end = Stb->cursor; }
int  LLMTextState::GetCursorPos() const              { return Stb->cursor; }
int  LLMTextState::GetSelectionStart() const         { return Stb->select_start; }
int  LLMTextState::GetSelectionEnd() const           { return Stb->select_end; }
void LLMTextState::SelectAll()                       { Stb->select_start = 0; Stb->cursor = Stb->select_end = TextLen; Stb->has_preferred_x = 0; }
void LLMTextState::ReloadUserBufAndSelectAll()       { WantReloadUserBuf = true; ReloadSelectionStart = 0; ReloadSelectionEnd = INT_MAX; }
void LLMTextState::ReloadUserBufAndKeepSelection()   { WantReloadUserBuf = true; ReloadSelectionStart = Stb->select_start; ReloadSelectionEnd = Stb->select_end; }
void LLMTextState::ReloadUserBufAndMoveToEnd()       { WantReloadUserBuf = true; ReloadSelectionStart = ReloadSelectionEnd = INT_MAX; }

static void InputTextReconcileUndoState(LLMTextState* state, const char* old_buf, int old_length, const char* new_buf, int new_length)
{
    const int shorter_length = ImMin(old_length, new_length);
    int first_diff;
    for (first_diff = 0; first_diff < shorter_length; first_diff++)
        if (old_buf[first_diff] != new_buf[first_diff])
            break;
    if (first_diff == old_length && first_diff == new_length)
        return;

    int old_last_diff = old_length   - 1;
    int new_last_diff = new_length - 1;
    for (; old_last_diff >= first_diff && new_last_diff >= first_diff; old_last_diff--, new_last_diff--)
        if (old_buf[old_last_diff] != new_buf[new_last_diff])
            break;

    const int insert_len = new_last_diff - first_diff + 1;
    const int delete_len = old_last_diff - first_diff + 1;
    if (insert_len > 0 || delete_len > 0)
        if (IMSTB_TEXTEDIT_CHARTYPE* p = LLMStb::stb_text_createundo(&state->Stb->undostate, first_diff, delete_len, insert_len))
            for (int i = 0; i < delete_len; i++)
                p[i] = old_buf[first_diff + i];
}

// Return false to discard a character.
static bool InputTextFilterCharacter(ImGuiContext* ctx, unsigned int* p_char, ImGuiInputTextFlags flags, ImGuiInputTextCallback callback, void* user_data, bool input_source_is_clipboard)
{
    unsigned int c = *p_char;

    // Filter non-printable (NB: isprint is unreliable! see #2467)
    bool apply_named_filters = true;
    if (c < 0x20)
    {
        bool pass = false;
        pass |= (c == '\n') && (flags & ImGuiInputTextFlags_Multiline) != 0; // Note that an Enter KEY will emit \r and be ignored (we poll for KEY in InputText() code)
        pass |= (c == '\t') && (flags & ImGuiInputTextFlags_AllowTabInput) != 0;
        if (!pass)
            return false;
        apply_named_filters = false; // Override named filters below so newline and tabs can still be inserted.
    }

    if (input_source_is_clipboard == false)
    {
        // We ignore Ascii representation of delete (emitted from Backspace on OSX, see #2578, #2817)
        if (c == 127)
            return false;

        // Filter private Unicode range. GLFW on OSX seems to send private characters for special keys like arrow keys (FIXME)
        if (c >= 0xE000 && c <= 0xF8FF)
            return false;
    }

    // Filter Unicode ranges we are not handling in this build
    if (c > IM_UNICODE_CODEPOINT_MAX)
        return false;

    // Generic named filters
    if (apply_named_filters && (flags & (ImGuiInputTextFlags_CharsDecimal | ImGuiInputTextFlags_CharsHexadecimal | ImGuiInputTextFlags_CharsUppercase | ImGuiInputTextFlags_CharsNoBlank | ImGuiInputTextFlags_CharsScientific | (ImGuiInputTextFlags)ImGuiInputTextFlags_LocalizeDecimalPoint)))
    {
        // The libc allows overriding locale, with e.g. 'setlocale(LC_NUMERIC, "de_DE.UTF-8");' which affect the output/input of printf/scanf to use e.g. ',' instead of '.'.
        // The standard mandate that programs starts in the "C" locale where the decimal point is '.'.
        // We don't really intend to provide widespread support for it, but out of empathy for people stuck with using odd API, we support the bare minimum aka overriding the decimal point.
        // Change the default decimal_point with:
        //   ImGui::GetPlatformIO()->Platform_LocaleDecimalPoint = *localeconv()->decimal_point;
        // Users of non-default decimal point (in particular ',') may be affected by word-selection logic (is_word_boundary_from_right/is_word_boundary_from_left) functions.
        ImGuiContext& g = *ctx;
        const unsigned c_decimal_point = (unsigned int)g.PlatformIO.Platform_LocaleDecimalPoint;
        if (flags & (ImGuiInputTextFlags_CharsDecimal | ImGuiInputTextFlags_CharsScientific | (ImGuiInputTextFlags)ImGuiInputTextFlags_LocalizeDecimalPoint))
            if (c == '.' || c == ',')
                c = c_decimal_point;

        // Full-width -> half-width conversion for numeric fields (https://en.wikipedia.org/wiki/Halfwidth_and_Fullwidth_Forms_(Unicode_block)
        // While this is mostly convenient, this has the side-effect for uninformed users accidentally inputting full-width characters that they may
        // scratch their head as to why it works in numerical fields vs in generic text fields it would require support in the font.
        if (flags & (ImGuiInputTextFlags_CharsDecimal | ImGuiInputTextFlags_CharsScientific | ImGuiInputTextFlags_CharsHexadecimal))
            if (c >= 0xFF01 && c <= 0xFF5E)
                c = c - 0xFF01 + 0x21;

        // Allow 0-9 . - + * /
        if (flags & ImGuiInputTextFlags_CharsDecimal)
            if (!(c >= '0' && c <= '9') && (c != c_decimal_point) && (c != '-') && (c != '+') && (c != '*') && (c != '/'))
                return false;

        // Allow 0-9 . - + * / e E
        if (flags & ImGuiInputTextFlags_CharsScientific)
            if (!(c >= '0' && c <= '9') && (c != c_decimal_point) && (c != '-') && (c != '+') && (c != '*') && (c != '/') && (c != 'e') && (c != 'E'))
                return false;

        // Allow 0-9 a-F A-F
        if (flags & ImGuiInputTextFlags_CharsHexadecimal)
            if (!(c >= '0' && c <= '9') && !(c >= 'a' && c <= 'f') && !(c >= 'A' && c <= 'F'))
                return false;

        // Turn a-z into A-Z
        if (flags & ImGuiInputTextFlags_CharsUppercase)
            if (c >= 'a' && c <= 'z')
                c += (unsigned int)('A' - 'a');

        if (flags & ImGuiInputTextFlags_CharsNoBlank)
            if (ImCharIsBlankW(c))
                return false;

        *p_char = c;
    }

    // Custom callback filter
    if (flags & ImGuiInputTextFlags_CallbackCharFilter)
    {
        ImGuiContext& g = *GImGui;
        ImGuiInputTextCallbackData callback_data;
        callback_data.Ctx = &g;
        callback_data.EventFlag = ImGuiInputTextFlags_CallbackCharFilter;
        callback_data.EventChar = (ImWchar)c;
        callback_data.Flags = flags;
        callback_data.UserData = user_data;
        if (callback(&callback_data) != 0)
            return false;
        *p_char = callback_data.EventChar;
        if (!callback_data.EventChar)
            return false;
    }

    return true;
}

bool CEditor::EditorWidget(const char* label, const char* hint, char* buf, int buf_size, const ImVec2& size_arg, ImGuiInputTextFlags flags)
{
	ImGuiInputTextCallback callback = NULL;
	void *callback_user_data = NULL;
	
    ImGuiWindow* window = GetCurrentWindow();
    if (window->SkipItems)
        return false;

    IM_ASSERT(buf != NULL && buf_size >= 0);
    IM_ASSERT(!((flags & ImGuiInputTextFlags_CallbackHistory) && (flags & ImGuiInputTextFlags_Multiline)));        // Can't use both together (they both use up/down keys)
    IM_ASSERT(!((flags & ImGuiInputTextFlags_CallbackCompletion) && (flags & ImGuiInputTextFlags_AllowTabInput))); // Can't use both together (they both use tab key)
    IM_ASSERT(!((flags & ImGuiInputTextFlags_ElideLeft) && (flags & ImGuiInputTextFlags_Multiline)));               // Multiline will not work with left-trimming

    ImGuiContext& g = *GImGui;
    ImGuiIO& io = g.IO;
    const ImGuiStyle& style = g.Style;

	int line_pitch = g.FontSize; // extra space per line
	int line_size = line_pitch + g.FontSize;

    const bool RENDER_SELECTION_WHEN_INACTIVE = false;
    const bool is_multiline = (flags & ImGuiInputTextFlags_Multiline) != 0;

    if (is_multiline) // Open group before calling GetID() because groups tracks id created within their scope (including the scrollbar)
        BeginGroup();
    const ImGuiID id = window->GetID(label);
    const ImVec2 label_size = CalcTextSize(label, NULL, true);
    const ImVec2 frame_size = CalcItemSize(size_arg, CalcItemWidth(), (is_multiline ? g.FontSize * 8.0f : label_size.y) + style.FramePadding.y * 2.0f); // Arbitrary default of 8 lines high for multi-line
    const ImVec2 total_size = ImVec2(frame_size.x + (label_size.x > 0.0f ? style.ItemInnerSpacing.x + label_size.x : 0.0f), frame_size.y);

    const ImRect frame_bb(window->DC.CursorPos, window->DC.CursorPos + frame_size);
    const ImRect total_bb(frame_bb.Min, frame_bb.Min + total_size);

    ImGuiWindow* draw_window = window;
    ImVec2 inner_size = frame_size;
    ImGuiLastItemData item_data_backup;
    if (is_multiline)
    {
        ImVec2 backup_pos = window->DC.CursorPos;
        ItemSize(total_bb, style.FramePadding.y);
        if (!ItemAdd(total_bb, id, &frame_bb, ImGuiItemFlags_Inputable))
        {
            EndGroup();
            return false;
        }
        item_data_backup = g.LastItemData;
        window->DC.CursorPos = backup_pos;

        // Prevent NavActivation from Tabbing when our widget accepts Tab inputs: this allows cycling through widgets without stopping.
        if (g.NavActivateId == id && (g.NavActivateFlags & ImGuiActivateFlags_FromTabbing) && (flags & ImGuiInputTextFlags_AllowTabInput))
            g.NavActivateId = 0;

        // Prevent NavActivate reactivating in BeginChild() when we are already active.
        const ImGuiID backup_activate_id = g.NavActivateId;
        if (g.ActiveId == id) // Prevent reactivation
            g.NavActivateId = 0;

        // We reproduce the contents of BeginChildFrame() in order to provide 'label' so our window internal data are easier to read/debug.
        PushStyleColor(ImGuiCol_ChildBg, style.Colors[ImGuiCol_FrameBg]);
        PushStyleVar(ImGuiStyleVar_ChildRounding, style.FrameRounding);
        PushStyleVar(ImGuiStyleVar_ChildBorderSize, style.FrameBorderSize);
        PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0)); // Ensure no clip rect so mouse hover can reach FramePadding edges
        bool child_visible = BeginChildEx(label, id, frame_bb.GetSize(), ImGuiChildFlags_Borders, ImGuiWindowFlags_NoMove);
        g.NavActivateId = backup_activate_id;
        PopStyleVar(3);
        PopStyleColor();
        if (!child_visible)
        {
            EndChild();
            EndGroup();
            return false;
        }
        draw_window = g.CurrentWindow; // Child window
        draw_window->DC.NavLayersActiveMaskNext |= (1 << draw_window->DC.NavLayerCurrent); // This is to ensure that EndChild() will display a navigation highlight so we can "enter" into it.
        draw_window->DC.CursorPos += style.FramePadding;
        inner_size.x -= draw_window->ScrollbarSizes.x;
    }
    else
    {
        // Support for internal ImGuiInputTextFlags_MergedItem flag, which could be redesigned as an ItemFlags if needed (with test performed in ItemAdd)
        ItemSize(total_bb, style.FramePadding.y);
        if (!(flags & ImGuiInputTextFlags_MergedItem))
            if (!ItemAdd(total_bb, id, &frame_bb, ImGuiItemFlags_Inputable))
                return false;
    }

    // Ensure mouse cursor is set even after switching to keyboard/gamepad mode. May generalize further? (#6417)
    bool hovered = ItemHoverable(frame_bb, id, g.LastItemData.ItemFlags | ImGuiItemFlags_NoNavDisableMouseHover);
    if (hovered)
        SetMouseCursor(ImGuiMouseCursor_TextInput);
    if (hovered && g.NavHighlightItemUnderNav)
        hovered = false;

    // Load our private text state.
    LLMTextState* state = &llmst;
	
	// in case we don't find it later
	state->current_tok = NULL;

    if (g.LastItemData.ItemFlags & ImGuiItemFlags_ReadOnly)
        flags |= ImGuiInputTextFlags_ReadOnly;
    const bool is_readonly = (flags & ImGuiInputTextFlags_ReadOnly) != 0;
    const bool is_password = (flags & ImGuiInputTextFlags_Password) != 0;
    const bool is_undoable = (flags & ImGuiInputTextFlags_NoUndoRedo) == 0;
    const bool is_resizable = (flags & ImGuiInputTextFlags_CallbackResize) != 0;
    if (is_resizable)
        IM_ASSERT(callback != NULL); // Must provide a callback if you set the ImGuiInputTextFlags_CallbackResize flag!

    const bool input_requested_by_nav = (g.ActiveId != id) && ((g.NavActivateId == id) && ((g.NavActivateFlags & ImGuiActivateFlags_PreferInput) || (g.NavInputSource == ImGuiInputSource_Keyboard)));

    const bool user_clicked = hovered && io.MouseClicked[0];
    const bool user_scroll_finish = is_multiline && state != NULL && g.ActiveId == 0 && g.ActiveIdPreviousFrame == GetWindowScrollbarID(draw_window, ImGuiAxis_Y);
    const bool user_scroll_active = is_multiline && state != NULL && g.ActiveId == GetWindowScrollbarID(draw_window, ImGuiAxis_Y);
    bool clear_active_id = false;
    bool select_all = false;

    float scroll_y = is_multiline ? draw_window->Scroll.y : FLT_MAX;

    const bool init_reload_from_user_buf = (state != NULL && state->WantReloadUserBuf);
    const bool init_changed_specs = (state != NULL && state->Stb->single_line != !is_multiline); // state != NULL means its our state.
    const bool init_make_active = (user_clicked || user_scroll_finish || input_requested_by_nav);
    const bool init_state = (init_make_active || user_scroll_active);
    if (false && init_reload_from_user_buf)
    {
        int new_len = (int)strlen(buf);
        IM_ASSERT(new_len + 1 <= buf_size && "Is your input buffer properly zero-terminated?");
        state->WantReloadUserBuf = false;
        InputTextReconcileUndoState(state, state->TextA.Data, state->TextLen, buf, new_len);
        state->TextA.resize(buf_size + 1); // we use +1 to make sure that .Data is always pointing to at least an empty string.
        state->TextLen = new_len;
        memcpy(state->TextA.Data, buf, state->TextLen + 1);
        state->Stb->select_start = state->ReloadSelectionStart;
        state->Stb->cursor = state->Stb->select_end = state->ReloadSelectionEnd;
        state->CursorClamp();
    }
    else if ((init_state && g.ActiveId != id) || init_changed_specs)
    {
        // Access state even if we don't own it yet.
        state = &llmst;
        state->CursorAnimReset();

        // Backup state of deactivating item so they'll have a chance to do a write to output buffer on the same frame they report IsItemDeactivatedAfterEdit (#4714)
        InputTextDeactivateHook(state->ID);

        // Take a copy of the initial buffer value.
        // From the moment we focused we are normally ignoring the content of 'buf' (unless we are in read-only mode)
        const int buf_len = (int)strlen(buf);
        IM_ASSERT(buf_len + 1 <= buf_size && "Is your input buffer properly zero-terminated?");
        state->TextToRevertTo.resize(buf_len + 1);    // UTF-8. we use +1 to make sure that .Data is always pointing to at least an empty string.
        memcpy(state->TextToRevertTo.Data, buf, buf_len + 1);

        // Preserve cursor position and undo/redo stack if we come back to same widget
        // FIXME: Since we reworked this on 2022/06, may want to differentiate recycle_cursor vs recycle_undostate?
        bool recycle_state = (state->ID == id && !init_changed_specs);
        if (recycle_state && (state->TextLen != buf_len || (state->TextA.Data == NULL || strncmp(state->TextA.Data, buf, buf_len) != 0)))
            recycle_state = false;

        // Start edition
        state->ID = id;
        state->TextLen = buf_len;
        if (!is_readonly)
        {
            state->TextA.resize(buf_size + 1); // we use +1 to make sure that .Data is always pointing to at least an empty string.
            memcpy(state->TextA.Data, buf, state->TextLen + 1);
        }

        // Find initial scroll position for right alignment
        state->Scroll = ImVec2(0.0f, 0.0f);
        if (flags & ImGuiInputTextFlags_ElideLeft)
            state->Scroll.x += ImMax(0.0f, CalcTextSize(buf).x - frame_size.x + style.FramePadding.x * 2.0f);

        // Recycle existing cursor/selection/undo stack but clamp position
        // Note a single mouse click will override the cursor/position immediately by calling stb_textedit_click handler.
        if (recycle_state)
            state->CursorClamp();
        else
            stb_textedit_initialize_state(state->Stb, !is_multiline);

        if (!is_multiline)
        {
            if (flags & ImGuiInputTextFlags_AutoSelectAll)
                select_all = true;
            if (input_requested_by_nav && (!recycle_state || !(g.NavActivateFlags & ImGuiActivateFlags_TryToPreserveState)))
                select_all = true;
            if (user_clicked && io.KeyCtrl)
                select_all = true;
        }

        if (flags & ImGuiInputTextFlags_AlwaysOverwrite)
            state->Stb->insert_mode = 1; // stb field name is indeed incorrect (see #2863)
    }

    const bool is_osx = io.ConfigMacOSXBehaviors;
    if (g.ActiveId != id && init_make_active)
    {
        IM_ASSERT(state && state->ID == id);
        SetActiveID(id, window);
        SetFocusID(id, window);
        FocusWindow(window);
    }
    if (g.ActiveId == id)
    {
        // Declare some inputs, the other are registered and polled via Shortcut() routing system.
        // FIXME: The reason we don't use Shortcut() is we would need a routing flag to specify multiple mods, or to all mods combinaison into individual shortcuts.
        const ImGuiKey always_owned_keys[] = { ImGuiKey_LeftArrow, ImGuiKey_RightArrow, ImGuiKey_Enter, ImGuiKey_KeypadEnter, ImGuiKey_Delete, ImGuiKey_Backspace, ImGuiKey_Home, ImGuiKey_End };
        for (ImGuiKey key : always_owned_keys)
            SetKeyOwner(key, id);
        if (user_clicked)
            SetKeyOwner(ImGuiKey_MouseLeft, id);
        g.ActiveIdUsingNavDirMask |= (1 << ImGuiDir_Left) | (1 << ImGuiDir_Right);
        if (is_multiline || (flags & ImGuiInputTextFlags_CallbackHistory))
        {
            g.ActiveIdUsingNavDirMask |= (1 << ImGuiDir_Up) | (1 << ImGuiDir_Down);
            SetKeyOwner(ImGuiKey_UpArrow, id);
            SetKeyOwner(ImGuiKey_DownArrow, id);
        }
        if (is_multiline)
        {
            SetKeyOwner(ImGuiKey_PageUp, id);
            SetKeyOwner(ImGuiKey_PageDown, id);
        }
        SetKeyOwner(ImGuiMod_Alt, id);

        // Expose scroll in a manner that is agnostic to us using a child window
        if (is_multiline && state != NULL)
            state->Scroll.y = draw_window->Scroll.y;

        // Read-only mode always ever read from source buffer. Refresh TextLen when active.
        if (is_readonly && state != NULL)
            state->TextLen = (int)strlen(buf);
        //if (is_readonly && state != NULL)
        //    state->TextA.clear(); // Uncomment to facilitate debugging, but we otherwise prefer to keep/amortize th allocation.
    }
    if (state != NULL)
        state->TextSrc = is_readonly ? buf : state->TextA.Data;

    // We have an edge case if ActiveId was set through another widget (e.g. widget being swapped), clear id immediately (don't wait until the end of the function)
    if (g.ActiveId == id && state == NULL)
        ClearActiveID();

    // Release focus when we click outside
    if (g.ActiveId == id && io.MouseClicked[0] && !init_state && !init_make_active) //-V560
        clear_active_id = true;

    // Lock the decision of whether we are going to take the path displaying the cursor or selection
    bool render_cursor = (g.ActiveId == id) || (state && user_scroll_active);
    bool render_selection = state && (state->HasSelection() || select_all) && (RENDER_SELECTION_WHEN_INACTIVE || render_cursor);
    bool value_changed = false;
    bool validated = false;

    // Select the buffer to render.
    const bool buf_display_from_state = (render_cursor || render_selection || g.ActiveId == id) && !is_readonly && state;
    const bool is_displaying_hint = (hint != NULL && (buf_display_from_state ? state->TextA.Data : buf)[0] == 0);

    // Password pushes a temporary font with only a fallback glyph
    if (is_password && !is_displaying_hint)
    {
        const ImFontGlyph* glyph = g.Font->FindGlyph('*');
        ImFont* password_font = &g.InputTextPasswordFont;
        password_font->FontSize = g.Font->FontSize;
        password_font->Scale = g.Font->Scale;
        password_font->Ascent = g.Font->Ascent;
        password_font->Descent = g.Font->Descent;
        password_font->ContainerAtlas = g.Font->ContainerAtlas;
        password_font->FallbackGlyph = glyph;
        password_font->FallbackAdvanceX = glyph->AdvanceX;
        IM_ASSERT(password_font->Glyphs.empty() && password_font->IndexAdvanceX.empty() && password_font->IndexLookup.empty());
        PushFont(password_font);
    }

    // Process mouse inputs and character inputs
    if (g.ActiveId == id)
    {
        IM_ASSERT(state != NULL);
        state->Edited = false;
        state->BufCapacity = buf_size;
        state->Flags = flags;

        // Although we are active we don't prevent mouse from hovering other elements unless we are interacting right now with the widget.
        // Down the line we should have a cleaner library-wide concept of Selected vs Active.
        g.ActiveIdAllowOverlap = !io.MouseDown[0];

        // Edit in progress
        const float mouse_x = (io.MousePos.x - frame_bb.Min.x - style.FramePadding.x) + state->Scroll.x;
        const float mouse_y = (is_multiline ? (io.MousePos.y - draw_window->DC.CursorPos.y)/2 : (g.FontSize * 0.5f));

        if (select_all)
        {
            state->SelectAll();
            state->SelectedAllMouseLock = true;
        }
        else if (hovered && io.MouseClickedCount[0] >= 2 && !io.KeyShift)
        {
            LLMStb::stb_textedit_click(state, state->Stb, mouse_x, mouse_y);
            const int multiclick_count = (io.MouseClickedCount[0] - 2);
            if ((multiclick_count % 2) == 0)
            {
                // Double-click: Select word
                // We always use the "Mac" word advance for double-click select vs CTRL+Right which use the platform dependent variant:
                // FIXME: There are likely many ways to improve this behavior, but there's no "right" behavior (depends on use-case, software, OS)
                const bool is_bol = (state->Stb->cursor == 0) || LLMStb::STB_TEXTEDIT_GETCHAR(state, state->Stb->cursor - 1) == '\n';
                if (STB_TEXT_HAS_SELECTION(state->Stb) || !is_bol)
                    state->OnKeyPressed(STB_TEXTEDIT_K_WORDLEFT);
                //state->OnKeyPressed(STB_TEXTEDIT_K_WORDRIGHT | STB_TEXTEDIT_K_SHIFT);
                if (!STB_TEXT_HAS_SELECTION(state->Stb))
                    LLMStb::stb_textedit_prep_selection_at_cursor(state->Stb);
                state->Stb->cursor = LLMStb::STB_TEXTEDIT_MOVEWORDRIGHT_MAC(state, state->Stb->cursor);
                state->Stb->select_end = state->Stb->cursor;
                LLMStb::stb_textedit_clamp(state, state->Stb);
            }
            else
            {
                // Triple-click: Select line
                const bool is_eol = LLMStb::STB_TEXTEDIT_GETCHAR(state, state->Stb->cursor) == '\n';
                state->OnKeyPressed(STB_TEXTEDIT_K_LINESTART);
                state->OnKeyPressed(STB_TEXTEDIT_K_LINEEND | STB_TEXTEDIT_K_SHIFT);
                state->OnKeyPressed(STB_TEXTEDIT_K_RIGHT | STB_TEXTEDIT_K_SHIFT);
                if (!is_eol && is_multiline)
                {
                    ImSwap(state->Stb->select_start, state->Stb->select_end);
                    state->Stb->cursor = state->Stb->select_end;
                }
                state->CursorFollow = false;
            }
            state->CursorAnimReset();
        }
        else if (io.MouseClicked[0] && !state->SelectedAllMouseLock)
        {
            if (hovered)
            {
                if (io.KeyShift)
                    stb_textedit_drag(state, state->Stb, mouse_x, mouse_y);
                else
                    stb_textedit_click(state, state->Stb, mouse_x, mouse_y);
                state->CursorAnimReset();
            }
        }
        else if (io.MouseDown[0] && !state->SelectedAllMouseLock && (io.MouseDelta.x != 0.0f || io.MouseDelta.y != 0.0f))
        {
            stb_textedit_drag(state, state->Stb, mouse_x, mouse_y);
            state->CursorAnimReset();
            state->CursorFollow = true;
        }
        if (state->SelectedAllMouseLock && !io.MouseDown[0])
            state->SelectedAllMouseLock = false;

        // We expect backends to emit a Tab key but some also emit a Tab character which we ignore (#2467, #1336)
        // (For Tab and Enter: Win32/SFML/Allegro are sending both keys and chars, GLFW and SDL are only sending keys. For Space they all send all threes)
        if ((flags & ImGuiInputTextFlags_AllowTabInput) && !is_readonly)
        {
            if (Shortcut(ImGuiKey_Tab, ImGuiInputFlags_Repeat, id))
            {
                unsigned int c = '\t'; // Insert TAB
                if (InputTextFilterCharacter(&g, &c, flags, callback, callback_user_data))
                    state->OnCharPressed(c);
            }
            // FIXME: Implement Shift+Tab
            /*
            if (Shortcut(ImGuiKey_Tab | ImGuiMod_Shift, ImGuiInputFlags_Repeat, id))
            {
            }
            */
        }

        // Process regular text input (before we check for Return because using some IME will effectively send a Return?)
        // We ignore CTRL inputs, but need to allow ALT+CTRL as some keyboards (e.g. German) use AltGR (which _is_ Alt+Ctrl) to input certain characters.
        const bool ignore_char_inputs = (io.KeyCtrl && !io.KeyAlt) || (is_osx && io.KeyCtrl);
        if (io.InputQueueCharacters.Size > 0)
        {
            if (!ignore_char_inputs && !is_readonly && !input_requested_by_nav)
                for (int n = 0; n < io.InputQueueCharacters.Size; n++)
                {
                    // Insert character if they pass filtering
                    unsigned int c = (unsigned int)io.InputQueueCharacters[n];
                    if (c == '\t') // Skip Tab, see above.
                        continue;
                    if (InputTextFilterCharacter(&g, &c, flags, callback, callback_user_data))
                        state->OnCharPressed(c);
                }

            // Consume characters
            io.InputQueueCharacters.resize(0);
        }
    }

    // Process other shortcuts/key-presses
    bool revert_edit = false;
    if (g.ActiveId == id && !g.ActiveIdIsJustActivated && !clear_active_id)
    {
        IM_ASSERT(state != NULL);

        const int row_count_per_page = ImMax((int)((inner_size.y - style.FramePadding.y) / line_size), 1);
        state->Stb->row_count_per_page = row_count_per_page;

        const int k_mask = (io.KeyShift ? STB_TEXTEDIT_K_SHIFT : 0);
        const bool is_wordmove_key_down = is_osx ? io.KeyAlt : io.KeyCtrl;                     // OS X style: Text editing cursor movement using Alt instead of Ctrl
        const bool is_startend_key_down = is_osx && io.KeyCtrl && !io.KeySuper && !io.KeyAlt;  // OS X style: Line/Text Start and End using Cmd+Arrows instead of Home/End

        // Using Shortcut() with ImGuiInputFlags_RouteFocused (default policy) to allow routing operations for other code (e.g. calling window trying to use CTRL+A and CTRL+B: formet would be handled by InputText)
        // Otherwise we could simply assume that we own the keys as we are active.
        const ImGuiInputFlags f_repeat = ImGuiInputFlags_Repeat;
        const bool is_cut   = (Shortcut(ImGuiMod_Ctrl | ImGuiKey_X, f_repeat, id) || Shortcut(ImGuiMod_Shift | ImGuiKey_Delete, f_repeat, id)) && !is_readonly && !is_password && (!is_multiline || state->HasSelection());
        const bool is_copy  = (Shortcut(ImGuiMod_Ctrl | ImGuiKey_C, 0,        id) || Shortcut(ImGuiMod_Ctrl  | ImGuiKey_Insert, 0,        id)) && !is_password && (!is_multiline || state->HasSelection());
        const bool is_paste = (Shortcut(ImGuiMod_Ctrl | ImGuiKey_V, f_repeat, id) || Shortcut(ImGuiMod_Shift | ImGuiKey_Insert, f_repeat, id)) && !is_readonly;
        const bool is_undo  = (Shortcut(ImGuiMod_Ctrl | ImGuiKey_Z, f_repeat, id)) && !is_readonly && is_undoable;
        const bool is_redo =  (Shortcut(ImGuiMod_Ctrl | ImGuiKey_Y, f_repeat, id) || (is_osx && Shortcut(ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiKey_Z, f_repeat, id))) && !is_readonly && is_undoable;
        const bool is_select_all = Shortcut(ImGuiMod_Ctrl | ImGuiKey_A, 0, id);

        // We allow validate/cancel with Nav source (gamepad) to makes it easier to undo an accidental NavInput press with no keyboard wired, but otherwise it isn't very useful.
        const bool nav_gamepad_active = (io.ConfigFlags & ImGuiConfigFlags_NavEnableGamepad) != 0 && (io.BackendFlags & ImGuiBackendFlags_HasGamepad) != 0;
        const bool is_enter_pressed = IsKeyPressed(ImGuiKey_Enter, true) || IsKeyPressed(ImGuiKey_KeypadEnter, true);
        const bool is_gamepad_validate = nav_gamepad_active && (IsKeyPressed(ImGuiKey_NavGamepadActivate, false) || IsKeyPressed(ImGuiKey_NavGamepadInput, false));
        const bool is_cancel = Shortcut(ImGuiKey_Escape, f_repeat, id) || (nav_gamepad_active && Shortcut(ImGuiKey_NavGamepadCancel, f_repeat, id));

		if(Shortcut(ImGuiMod_Alt | ImGuiKey_LeftArrow, f_repeat, id)) {
			state->Stb->cursor = state->llm.alt_back(state->Stb->cursor);
			state->CursorFollow = true;
		} else
		if(Shortcut(ImGuiMod_Alt | ImGuiKey_RightArrow, f_repeat, id)) {
			state->Stb->cursor = state->llm.alt_commit(state->Stb->cursor);
			state->CursorFollow = true;
		} else
		if(Shortcut(ImGuiMod_Alt | ImGuiKey_UpArrow, f_repeat, id)) {
			state->llm.alt_prev(state->Stb->cursor);
			state->invalidate_predictions = true;
		} else
		if(Shortcut(ImGuiMod_Alt | ImGuiKey_DownArrow, f_repeat, id)) {
			state->llm.alt_next(state->Stb->cursor);
			state->invalidate_predictions = true;
		} else
        // FIXME: Should use more Shortcut() and reduce IsKeyPressed()+SetKeyOwner(), but requires modifiers combination to be taken account of.
        // FIXME-OSX: Missing support for Alt(option)+Right/Left = go to end of line, or next line if already in end of line.
        if (IsKeyPressed(ImGuiKey_LeftArrow))                        { state->OnKeyPressed((is_startend_key_down ? STB_TEXTEDIT_K_LINESTART : is_wordmove_key_down ? STB_TEXTEDIT_K_WORDLEFT : STB_TEXTEDIT_K_LEFT) | k_mask); }
        else if (IsKeyPressed(ImGuiKey_RightArrow))                  { state->OnKeyPressed((is_startend_key_down ? STB_TEXTEDIT_K_LINEEND : is_wordmove_key_down ? STB_TEXTEDIT_K_WORDRIGHT : STB_TEXTEDIT_K_RIGHT) | k_mask); }
        else if (IsKeyPressed(ImGuiKey_UpArrow) && is_multiline)     { if (io.KeyCtrl) SetScrollY(draw_window, ImMax(draw_window->Scroll.y - line_size, 0.0f)); else state->OnKeyPressed((is_startend_key_down ? STB_TEXTEDIT_K_TEXTSTART : STB_TEXTEDIT_K_UP) | k_mask); }
        else if (IsKeyPressed(ImGuiKey_DownArrow) && is_multiline)   { if (io.KeyCtrl) SetScrollY(draw_window, ImMin(draw_window->Scroll.y + line_size, GetScrollMaxY())); else state->OnKeyPressed((is_startend_key_down ? STB_TEXTEDIT_K_TEXTEND : STB_TEXTEDIT_K_DOWN) | k_mask); }
        else if (IsKeyPressed(ImGuiKey_PageUp) && is_multiline)      { state->OnKeyPressed(STB_TEXTEDIT_K_PGUP | k_mask); scroll_y -= row_count_per_page * line_size; }
        else if (IsKeyPressed(ImGuiKey_PageDown) && is_multiline)    { state->OnKeyPressed(STB_TEXTEDIT_K_PGDOWN | k_mask); scroll_y += row_count_per_page * line_size; }
        else if (IsKeyPressed(ImGuiKey_Home))                        { state->OnKeyPressed(io.KeyCtrl ? STB_TEXTEDIT_K_TEXTSTART | k_mask : STB_TEXTEDIT_K_LINESTART | k_mask); }
        else if (IsKeyPressed(ImGuiKey_End))                         { state->OnKeyPressed(io.KeyCtrl ? STB_TEXTEDIT_K_TEXTEND | k_mask : STB_TEXTEDIT_K_LINEEND | k_mask); }
        else if (IsKeyPressed(ImGuiKey_Delete) && !is_readonly && !is_cut)
        {
            if (!state->HasSelection())
            {
                // OSX doesn't seem to have Super+Delete to delete until end-of-line, so we don't emulate that (as opposed to Super+Backspace)
                if (is_wordmove_key_down)
                    state->OnKeyPressed(STB_TEXTEDIT_K_WORDRIGHT | STB_TEXTEDIT_K_SHIFT);
            }
            state->OnKeyPressed(STB_TEXTEDIT_K_DELETE | k_mask);
        }
        else if (IsKeyPressed(ImGuiKey_Backspace) && !is_readonly)
        {
            if (!state->HasSelection())
            {
                if (is_wordmove_key_down)
                    state->OnKeyPressed(STB_TEXTEDIT_K_WORDLEFT | STB_TEXTEDIT_K_SHIFT);
                else if (is_osx && io.KeyCtrl && !io.KeyAlt && !io.KeySuper)
                    state->OnKeyPressed(STB_TEXTEDIT_K_LINESTART | STB_TEXTEDIT_K_SHIFT);
            }
            state->OnKeyPressed(STB_TEXTEDIT_K_BACKSPACE | k_mask);
        }
        else if (is_enter_pressed || is_gamepad_validate)
        {
            // Determine if we turn Enter into a \n character
            bool ctrl_enter_for_new_line = (flags & ImGuiInputTextFlags_CtrlEnterForNewLine) != 0;
            if (!is_multiline || is_gamepad_validate || (ctrl_enter_for_new_line && !io.KeyCtrl) || (!ctrl_enter_for_new_line && io.KeyCtrl))
            {
                validated = true;
                if (io.ConfigInputTextEnterKeepActive && !is_multiline)
                    state->SelectAll(); // No need to scroll
                else
                    clear_active_id = true;
            }
            else if (!is_readonly)
            {
                unsigned int c = '\n'; // Insert new line
                if (InputTextFilterCharacter(&g, &c, flags, callback, callback_user_data))
                    state->OnCharPressed(c);
            }
        }
        else if (is_cancel)
        {
            if (flags & ImGuiInputTextFlags_EscapeClearsAll)
            {
                if (buf[0] != 0)
                {
                    revert_edit = true;
                }
                else
                {
                    render_cursor = render_selection = false;
                    clear_active_id = true;
                }
            }
            else
            {
                clear_active_id = revert_edit = true;
                render_cursor = render_selection = false;
            }
        }
        else if (is_undo || is_redo)
        {
            state->OnKeyPressed(is_undo ? STB_TEXTEDIT_K_UNDO : STB_TEXTEDIT_K_REDO);
            state->ClearSelection();
        }
        else if (is_select_all)
        {
            state->SelectAll();
            state->CursorFollow = true;
        }
        else if (is_cut || is_copy)
        {
            // Cut, Copy
            if (g.PlatformIO.Platform_SetClipboardTextFn != NULL)
            {
                // SetClipboardText() only takes null terminated strings + state->TextSrc may point to read-only user buffer, so we need to make a copy.
                const int ib = state->HasSelection() ? ImMin(state->Stb->select_start, state->Stb->select_end) : 0;
                const int ie = state->HasSelection() ? ImMax(state->Stb->select_start, state->Stb->select_end) : state->TextLen;
                g.TempBuffer.reserve(ie - ib + 1);
                memcpy(g.TempBuffer.Data, state->TextSrc + ib, ie - ib);
                g.TempBuffer.Data[ie - ib] = 0;
                SetClipboardText(g.TempBuffer.Data);
            }
            if (is_cut)
            {
                if (!state->HasSelection())
                    state->SelectAll();
                state->CursorFollow = true;
                stb_textedit_cut(state, state->Stb);
            }
        }
        else if (is_paste)
        {
            if (const char* clipboard = GetClipboardText())
            {
                // Filter pasted buffer
                const int clipboard_len = (int)strlen(clipboard);
                ImVector<char> clipboard_filtered;
                clipboard_filtered.reserve(clipboard_len + 1);
                for (const char* s = clipboard; *s != 0; )
                {
                    unsigned int c;
                    int in_len = ImTextCharFromUtf8(&c, s, NULL);
                    s += in_len;
                    if (!InputTextFilterCharacter(&g, &c, flags, callback, callback_user_data, true))
                        continue;
                    char c_utf8[5];
                    ImTextCharToUtf8(c_utf8, c);
                    int out_len = (int)strlen(c_utf8);
                    clipboard_filtered.resize(clipboard_filtered.Size + out_len);
                    memcpy(clipboard_filtered.Data + clipboard_filtered.Size - out_len, c_utf8, out_len);
                }
                if (clipboard_filtered.Size > 0) // If everything was filtered, ignore the pasting operation
                {
                    clipboard_filtered.push_back(0);
                    stb_textedit_paste(state, state->Stb, clipboard_filtered.Data, clipboard_filtered.Size - 1);
                    state->CursorFollow = true;
                }
            }
        }

        // Update render selection flag after events have been handled, so selection highlight can be displayed during the same frame.
        render_selection |= state->HasSelection() && (RENDER_SELECTION_WHEN_INACTIVE || render_cursor);
    }

    // Process callbacks and apply result back to user's buffer.
    const char* apply_new_text = NULL;
    int apply_new_text_length = 0;
    if (g.ActiveId == id)
    {
        IM_ASSERT(state != NULL);
        if (revert_edit && !is_readonly)
        {
            if (flags & ImGuiInputTextFlags_EscapeClearsAll)
            {
                // Clear input
                IM_ASSERT(buf[0] != 0);
                apply_new_text = "";
                apply_new_text_length = 0;
                value_changed = true;
                IMSTB_TEXTEDIT_CHARTYPE empty_string;
                stb_textedit_replace(state, state->Stb, &empty_string, 0);
            }
            else if (strcmp(buf, state->TextToRevertTo.Data) != 0)
            {
                apply_new_text = state->TextToRevertTo.Data;
                apply_new_text_length = state->TextToRevertTo.Size - 1;

                // Restore initial value. Only return true if restoring to the initial value changes the current buffer contents.
                // Push records into the undo stack so we can CTRL+Z the revert operation itself
                value_changed = true;
                stb_textedit_replace(state, state->Stb, state->TextToRevertTo.Data, state->TextToRevertTo.Size - 1);
            }
        }

        // FIXME-OPT: We always reapply the live buffer back to the input buffer before clearing ActiveId,
        // even though strictly speaking it wasn't modified on this frame. Should mark dirty state from the stb_textedit callbacks.
        // If we do that, need to ensure that as special case, 'validated == true' also writes back.
        // This also allows the user to use InputText() without maintaining any user-side storage.
        // (please note that if you use this property along ImGuiInputTextFlags_CallbackResize you can end up with your temporary string object
        // unnecessarily allocating once a frame, either store your string data, either if you don't then don't use ImGuiInputTextFlags_CallbackResize).
        const bool apply_edit_back_to_user_buffer = true;// !revert_edit || (validated && (flags & ImGuiInputTextFlags_EnterReturnsTrue) != 0);
        if (apply_edit_back_to_user_buffer)
        {
            // Apply current edited text immediately.
            // Note that as soon as the input box is active, the in-widget value gets priority over any underlying modification of the input buffer

            // User callback
            if ((flags & (ImGuiInputTextFlags_CallbackCompletion | ImGuiInputTextFlags_CallbackHistory | ImGuiInputTextFlags_CallbackEdit | ImGuiInputTextFlags_CallbackAlways)) != 0)
            {
                IM_ASSERT(callback != NULL);

                // The reason we specify the usage semantic (Completion/History) is that Completion needs to disable keyboard TABBING at the moment.
                ImGuiInputTextFlags event_flag = 0;
                ImGuiKey event_key = ImGuiKey_None;
                if ((flags & ImGuiInputTextFlags_CallbackCompletion) != 0 && Shortcut(ImGuiKey_Tab, 0, id))
                {
                    event_flag = ImGuiInputTextFlags_CallbackCompletion;
                    event_key = ImGuiKey_Tab;
                }
                else if ((flags & ImGuiInputTextFlags_CallbackHistory) != 0 && IsKeyPressed(ImGuiKey_UpArrow))
                {
                    event_flag = ImGuiInputTextFlags_CallbackHistory;
                    event_key = ImGuiKey_UpArrow;
                }
                else if ((flags & ImGuiInputTextFlags_CallbackHistory) != 0 && IsKeyPressed(ImGuiKey_DownArrow))
                {
                    event_flag = ImGuiInputTextFlags_CallbackHistory;
                    event_key = ImGuiKey_DownArrow;
                }
                else if ((flags & ImGuiInputTextFlags_CallbackEdit) && state->Edited)
                {
                    event_flag = ImGuiInputTextFlags_CallbackEdit;
                }
                else if (flags & ImGuiInputTextFlags_CallbackAlways)
                {
                    event_flag = ImGuiInputTextFlags_CallbackAlways;
                }

                if (event_flag)
                {
                    ImGuiInputTextCallbackData callback_data;
                    callback_data.Ctx = &g;
                    callback_data.EventFlag = event_flag;
                    callback_data.Flags = flags;
                    callback_data.UserData = callback_user_data;

                    // FIXME-OPT: Undo stack reconcile needs a backup of the data until we rework API, see #7925
                    char* callback_buf = is_readonly ? buf : state->TextA.Data;
                    IM_ASSERT(callback_buf == state->TextSrc);
                    state->CallbackTextBackup.resize(state->TextLen + 1);
                    memcpy(state->CallbackTextBackup.Data, callback_buf, state->TextLen + 1);

                    callback_data.EventKey = event_key;
                    callback_data.Buf = callback_buf;
                    callback_data.BufTextLen = state->TextLen;
                    callback_data.BufSize = state->BufCapacity;
                    callback_data.BufDirty = false;

                    const int utf8_cursor_pos = callback_data.CursorPos = state->Stb->cursor;
                    const int utf8_selection_start = callback_data.SelectionStart = state->Stb->select_start;
                    const int utf8_selection_end = callback_data.SelectionEnd = state->Stb->select_end;

                    // Call user code
                    callback(&callback_data);

                    // Read back what user may have modified
                    callback_buf = is_readonly ? buf : state->TextA.Data; // Pointer may have been invalidated by a resize callback
                    IM_ASSERT(callback_data.Buf == callback_buf);         // Invalid to modify those fields
                    IM_ASSERT(callback_data.BufSize == state->BufCapacity);
                    IM_ASSERT(callback_data.Flags == flags);
                    const bool buf_dirty = callback_data.BufDirty;
                    if (callback_data.CursorPos != utf8_cursor_pos || buf_dirty)            { state->Stb->cursor = callback_data.CursorPos; state->CursorFollow = true; }
                    if (callback_data.SelectionStart != utf8_selection_start || buf_dirty)  { state->Stb->select_start = (callback_data.SelectionStart == callback_data.CursorPos) ? state->Stb->cursor : callback_data.SelectionStart; }
                    if (callback_data.SelectionEnd != utf8_selection_end || buf_dirty)      { state->Stb->select_end = (callback_data.SelectionEnd == callback_data.SelectionStart) ? state->Stb->select_start : callback_data.SelectionEnd; }
                    if (buf_dirty)
                    {
                        // Callback may update buffer and thus set buf_dirty even in read-only mode.
                        IM_ASSERT(callback_data.BufTextLen == (int)strlen(callback_data.Buf)); // You need to maintain BufTextLen if you change the text!
                        InputTextReconcileUndoState(state, state->CallbackTextBackup.Data, state->CallbackTextBackup.Size - 1, callback_data.Buf, callback_data.BufTextLen);
                        state->TextLen = callback_data.BufTextLen;  // Assume correct length and valid UTF-8 from user, saves us an extra strlen()
                        state->CursorAnimReset();
                    }
                }
            }

            // Will copy result string if modified
            if (!is_readonly && strcmp(state->TextSrc, buf) != 0)
            {
                apply_new_text = state->TextSrc;
                apply_new_text_length = state->TextLen;
                value_changed = true;
            }
        }
    }

    // Handle reapplying final data on deactivation (see InputTextDeactivateHook() for details)
    if (g.InputTextDeactivatedState.ID == id)
    {
        if (g.ActiveId != id && IsItemDeactivatedAfterEdit() && !is_readonly && strcmp(g.InputTextDeactivatedState.TextA.Data, buf) != 0)
        {
            apply_new_text = g.InputTextDeactivatedState.TextA.Data;
            apply_new_text_length = g.InputTextDeactivatedState.TextA.Size - 1;
            value_changed = true;
            //IMGUI_DEBUG_LOG("InputText(): apply Deactivated data for 0x%08X: \"%.*s\".\n", id, apply_new_text_length, apply_new_text);
        }
        g.InputTextDeactivatedState.ID = 0;
    }

    // Copy result to user buffer. This can currently only happen when (g.ActiveId == id)
    if (apply_new_text != NULL)
    {
        //// We cannot test for 'backup_current_text_length != apply_new_text_length' here because we have no guarantee that the size
        //// of our owned buffer matches the size of the string object held by the user, and by design we allow InputText() to be used
        //// without any storage on user's side.
        IM_ASSERT(apply_new_text_length >= 0);
        if (is_resizable)
        {
            ImGuiInputTextCallbackData callback_data;
            callback_data.Ctx = &g;
            callback_data.EventFlag = ImGuiInputTextFlags_CallbackResize;
            callback_data.Flags = flags;
            callback_data.Buf = buf;
            callback_data.BufTextLen = apply_new_text_length;
            callback_data.BufSize = ImMax(buf_size, apply_new_text_length + 1);
            callback_data.UserData = callback_user_data;
            callback(&callback_data);
            buf = callback_data.Buf;
            buf_size = callback_data.BufSize;
            apply_new_text_length = ImMin(callback_data.BufTextLen, buf_size - 1);
            IM_ASSERT(apply_new_text_length <= buf_size);
        }
        //IMGUI_DEBUG_PRINT("InputText(\"%s\"): apply_new_text length %d\n", label, apply_new_text_length);

        // If the underlying buffer resize was denied or not carried to the next frame, apply_new_text_length+1 may be >= buf_size.
        ImStrncpy(buf, apply_new_text, ImMin(apply_new_text_length + 1, buf_size));
    }

    // Release active ID at the end of the function (so e.g. pressing Return still does a final application of the value)
    // Otherwise request text input ahead for next frame.
    if (g.ActiveId == id && clear_active_id)
        ClearActiveID();
    else if (g.ActiveId == id)
        g.WantTextInputNextFrame = 1;

    // Render frame
    if (!is_multiline)
    {
        RenderNavCursor(frame_bb, id);
        RenderFrame(frame_bb.Min, frame_bb.Max, GetColorU32(ImGuiCol_FrameBg), true, style.FrameRounding);
    }

    const ImVec4 clip_rect(frame_bb.Min.x, frame_bb.Min.y, frame_bb.Min.x + inner_size.x, frame_bb.Min.y + inner_size.y); // Not using frame_bb.Max because we have adjusted size
    ImVec2 draw_pos = is_multiline ? draw_window->DC.CursorPos : frame_bb.Min + style.FramePadding;
    ImVec2 text_size(0.0f, 0.0f);

    // Set upper limit of single-line InputTextEx() at 2 million characters strings. The current pathological worst case is a long line
    // without any carriage return, which would makes ImFont::RenderText() reserve too many vertices and probably crash. Avoid it altogether.
    // Note that we only use this limit on single-line InputText(), so a pathologically large line on a InputTextMultiline() would still crash.
    const int buf_display_max_length = 2 * 1024 * 1024;
    const char* buf_display = buf_display_from_state ? state->TextA.Data : buf; //-V595
    const char* buf_display_end = NULL; // We have specialized paths below for setting the length
    if (is_displaying_hint)
    {
        buf_display = hint;
        buf_display_end = hint + strlen(hint);
    }

    // Render text. We currently only render selection when the widget is active or while scrolling.
    // FIXME: We could remove the '&& render_cursor' to keep rendering selection when inactive.
    if (true || render_cursor || render_selection)
    {
        IM_ASSERT(state != NULL);
        if (!is_displaying_hint)
            buf_display_end = buf_display + state->TextLen;

        // Render text (with cursor and selection)
        // This is going to be messy. We need to:
        // - Display the text (this alone can be more easily clipped)
        // - Handle scrolling, highlight selection, display cursor (those all requires some form of 1d->2d cursor position calculation)
        // - Measure text height (for scrollbar)
        // We are attempting to do most of that in **one main pass** to minimize the computation cost (non-negligible for large amount of text) + 2nd pass for selection rendering (we could merge them by an extra refactoring effort)
        // FIXME: This should occur on buf_display but we'd need to maintain cursor/select_start/select_end for UTF-8.
        const char* text_begin = buf_display;
        const char* text_end = text_begin + state->TextLen;
        ImVec2 cursor_offset, select_start_offset;

        {
            // Find lines numbers straddling cursor and selection min position
            int cursor_line_no = render_cursor ? -1 : -1000;
            int selmin_line_no = render_selection ? -1 : -1000;
            const char* cursor_ptr = render_cursor ? text_begin + state->Stb->cursor : NULL;
            const char* selmin_ptr = render_selection ? text_begin + ImMin(state->Stb->select_start, state->Stb->select_end) : NULL;

            // Count lines and find line number for cursor and selection ends
            int line_count = 1;
            if (is_multiline)
            {
                for (const char* s = text_begin; (s = (const char*)memchr(s, '\n', (size_t)(text_end - s))) != NULL; s++)
                {
                    if (cursor_line_no == -1 && s >= cursor_ptr) { cursor_line_no = line_count; }
                    if (selmin_line_no == -1 && s >= selmin_ptr) { selmin_line_no = line_count; }
                    line_count++;
                }
            }
            if (cursor_line_no == -1)
                cursor_line_no = line_count;
            if (selmin_line_no == -1)
                selmin_line_no = line_count;

            // Calculate 2d position by finding the beginning of the line and measuring distance
            cursor_offset.x = InputTextCalcTextSize(&g, ImStrbol(cursor_ptr, text_begin), cursor_ptr).x;
            cursor_offset.y = cursor_line_no * line_size;
            if (selmin_line_no >= 0)
            {
                select_start_offset.x = InputTextCalcTextSize(&g, ImStrbol(selmin_ptr, text_begin), selmin_ptr).x;
                select_start_offset.y = selmin_line_no * line_size;
            }

            // Store text height (note that we haven't calculated text width at all, see GitHub issues #383, #1224)
            if (is_multiline)
                text_size = ImVec2(inner_size.x, line_count * line_size);
        }

        // Scroll
        if (render_cursor && state->CursorFollow)
        {
            // Horizontal scroll in chunks of quarter width
            if (!(flags & ImGuiInputTextFlags_NoHorizontalScroll))
            {
                const float scroll_increment_x = inner_size.x * 0.25f;
                const float visible_width = inner_size.x - style.FramePadding.x;
                if (cursor_offset.x < state->Scroll.x)
                    state->Scroll.x = IM_TRUNC(ImMax(0.0f, cursor_offset.x - scroll_increment_x));
                else if (cursor_offset.x - visible_width >= state->Scroll.x)
                    state->Scroll.x = IM_TRUNC(cursor_offset.x - visible_width + scroll_increment_x);
            }
            else
            {
                state->Scroll.y = 0.0f;
            }

            // Vertical scroll
            if (is_multiline)
            {
                // Test if cursor is vertically visible
                if (cursor_offset.y - line_size < scroll_y)
                    scroll_y = ImMax(0.0f, cursor_offset.y - line_size);
                else if (cursor_offset.y - (inner_size.y - style.FramePadding.y * 2.0f) >= scroll_y)
                    scroll_y = cursor_offset.y - inner_size.y + style.FramePadding.y * 2.0f;
                const float scroll_max_y = ImMax((text_size.y + style.FramePadding.y * 2.0f) - inner_size.y, 0.0f);
                scroll_y = ImClamp(scroll_y, 0.0f, scroll_max_y);
                draw_pos.y += (draw_window->Scroll.y - scroll_y);   // Manipulate cursor pos immediately avoid a frame of lag
                draw_window->Scroll.y = scroll_y;
            }

            state->CursorFollow = false;
        }

        // Draw selection
        const ImVec2 draw_scroll = ImVec2(state->Scroll.x, 0.0f);
        if (render_selection)
        {
            const char* text_selected_begin = text_begin + ImMin(state->Stb->select_start, state->Stb->select_end);
            const char* text_selected_end = text_begin + ImMax(state->Stb->select_start, state->Stb->select_end);

            ImU32 bg_color = GetColorU32(ImGuiCol_TextSelectedBg, render_cursor ? 1.0f : 0.6f); // FIXME: current code flow mandate that render_cursor is always true here, we are leaving the transparent one for tests.
            float bg_offy_up = is_multiline ? 0.0f : -1.0f;    // FIXME: those offsets should be part of the style? they don't play so well with multi-line selection.
            float bg_offy_dn = is_multiline ? 0.0f : 2.0f;
            ImVec2 rect_pos = draw_pos + select_start_offset - draw_scroll;
            for (const char* p = text_selected_begin; p < text_selected_end; )
            {
                if (rect_pos.y > clip_rect.w + g.FontSize)
                    break;
                if (rect_pos.y < clip_rect.y)
                {
                    p = (const char*)memchr((void*)p, '\n', text_selected_end - p);
                    p = p ? p + 1 : text_selected_end;
                }
                else
                {
                    ImVec2 rect_size = InputTextCalcTextSize(&g, p, text_selected_end, &p, NULL, true);
                    if (rect_size.x <= 0.0f) rect_size.x = IM_TRUNC(g.Font->GetCharAdvance((ImWchar)' ') * 0.50f); // So we can see selected empty lines
                    ImRect rect(rect_pos + ImVec2(0.0f, bg_offy_up - g.FontSize), rect_pos + ImVec2(rect_size.x, bg_offy_dn));
                    rect.ClipWith(clip_rect);
                    if (rect.Overlaps(clip_rect))
                        draw_window->DrawList->AddRectFilled(rect.Min, rect.Max, bg_color);
                    rect_pos.x = draw_pos.x - draw_scroll.x;
                }
                rect_pos.y += line_size;
            }
        }
		
		// Mark background up with LLM tree state
		{
			TTE *cur = &state->llm.root;
			int offs = 0;
			int line_offs = 1;
			ImVec2 pos;
			bool cursor_found = false;
			while(cur) {
				pos.x = InputTextCalcTextSize(&g, ImStrbol(text_begin + offs, text_begin), text_begin + offs).x;
                pos.y = line_offs * line_size;
				
				ImVec2 rect_pos = draw_pos + pos - draw_scroll;
				// mark positions that have a snapshot, for debug purposes
				if(cur->ctx_snapshot) {
					draw_window->DrawList->AddCircleFilled(rect_pos + ImVec2(0.0, -g.FontSize), 2.5, ImColor(0.0f, 0.8f, 0.0f, 1.0f), 4);
				}
                // mark positions with more than one actualized descendant
                if(cur->parent && cur->parent->children.size()>1) {
                    int cnt=0;
                    for(auto &c : cur->parent->children) cnt += (c->is_accepted);
                    if(cnt>1)
                        draw_window->DrawList->AddCircle(rect_pos + ImVec2(0.0, -g.FontSize), 5.0, ImColor(0.0f, 0.0f, 1.0f, 0.5f), 3);
                }
				
				const char *tok_begin = text_begin + offs;
				const char *tok_end = tok_begin + cur->str_size;
				
				if(!cursor_found && state->Stb->cursor <= offs) {
					cursor_found = true;
					TTE *parent = cur->parent; if(!parent) parent=cur;
					
					// maybe request new alternative predictions here
					if(state->last_tok != parent) {
						state->llm.req_alts_at_pos(offs);
						state->invalidate_predictions = true;
					}
					state->current_tok = state->last_tok = parent;
					
					// render predictions
					if(state->invalidate_predictions) {
						if(parent->sel>0) {
							state->above = state->llm.render(parent->children[parent->sel-1], state->llm.predict_alt,true);
							std::replace( state->above.begin(), state->above.end(), '\n', '\\');
						} else state->above = "";
						if(parent->children.size()>(parent->sel)) {
							state->selected = state->llm.render(parent->children[parent->sel], state->llm.predict_main,true);
							std::replace( state->selected.begin(), state->selected.end(), '\n', '\\');
						} else state->selected = "";
						if(parent->children.size()>(parent->sel+1)) {
							state->below = state->llm.render(parent->children[parent->sel+1], state->llm.predict_alt,true);
							std::replace( state->below.begin(), state->below.end(), '\n', '\\');
						} else {
							state->below = "";
						}
						state->invalidate_predictions = false;
					}
					//int delta;
					//state->llm.get_alts_at_pos(offs, state->above, state->selected, state->below, delta);
					
					ImU32 clr_pred = GetColorU32(ImGuiCol_TextDisabled);
					draw_window->DrawList->AddText(g.Font, g.FontSize, rect_pos + ImVec2(0, -2*g.FontSize), clr_pred, state->above.c_str(), NULL, 0.0f, NULL);
					draw_window->DrawList->AddText(g.Font, g.FontSize, rect_pos + ImVec2(0, -g.FontSize), clr_pred, state->selected.c_str(), NULL, 0.0f, NULL);
					draw_window->DrawList->AddText(g.Font, g.FontSize, rect_pos, clr_pred, state->below.c_str(), NULL, 0.0f, NULL);
				}
				
				// go no further if this token is only predicted
				if(!cur->is_accepted) break;
				
				for (const char* p = tok_begin; p < tok_end; )
				{
					if (rect_pos.y > clip_rect.w + g.FontSize)
						break;
					if (rect_pos.y < clip_rect.y)
					{
						p = (const char*)memchr((void*)p, '\n', tok_end - p);
						p = p ? p + 1 : tok_end;
					}
					else
					{
						ImVec2 rect_size = InputTextCalcTextSize(&g, p, tok_end, &p, NULL, true);
						if (rect_size.x <= 0.0f) rect_size.x = IM_TRUNC(g.Font->GetCharAdvance((ImWchar)' ') * 0.50f); // So we can see selected empty lines
						ImRect rect(rect_pos + ImVec2(0.0f, 0.0f - g.FontSize), rect_pos + ImVec2(rect_size.x, 0.0f));
						rect.ClipWith(clip_rect);
						if (rect.Overlaps(clip_rect)) {
							ImColor logit_c;
							if(cur->has_logit) {
								float logit_diff = cur->logit - cur->max_logit;
								float logit_scaled = -((logit_diff)/1.6);
								logit_c = ImColor(1.0f, 0.0f, 0.0f, 0.1f*logit_scaled);
							} else {
								logit_c = ImColor(0.5f,0.5f,0.5f,0.5f);
							}
							draw_window->DrawList->AddRectFilled(rect.Min, rect.Max, logit_c);
						}
						rect_pos.x = draw_pos.x - draw_scroll.x;
					}
					rect_pos.y += line_size;
				}
				
				// advance line count by #lines in token
				line_offs += count(cur->str.begin(), cur->str.end(), '\n');
				// next token
				if(!cur->children.size()) cur=NULL;
				else {
					offs += cur->str_size;
					cur=cur->children[cur->sel];
				}
			}
		}

        // We test for 'buf_display_max_length' as a way to avoid some pathological cases (e.g. single-line 1 MB string) which would make ImDrawList crash.
        // FIXME-OPT: Multiline could submit a smaller amount of contents to AddText() since we already iterated through it.
        if (is_multiline || (buf_display_end - buf_display) < buf_display_max_length)
        {
            ImU32 col = GetColorU32(is_displaying_hint ? ImGuiCol_TextDisabled : ImGuiCol_Text);
			int offs = 0;
			for (const char* s = buf_display, *snext; s != NULL && s < buf_display_end; s = snext)
			{
				snext = (const char*)memchr(s, '\n', (size_t)(text_end - s));
				
				draw_window->DrawList->AddText(g.Font, g.FontSize, draw_pos - draw_scroll + ImVec2(0, g.FontSize + offs * line_size), col, s, snext, 0.0f, is_multiline ? NULL : &clip_rect);
				
				if(snext!=NULL) snext++; // advance past \n
				
				++offs;
			}
        }

        // Draw blinking cursor
        if (render_cursor)
        {
            state->CursorAnim += io.DeltaTime;
            bool cursor_is_visible = (!g.IO.ConfigInputTextCursorBlink) || (state->CursorAnim <= 0.0f) || ImFmod(state->CursorAnim, 1.20f) <= 0.80f;
            ImVec2 cursor_screen_pos = ImTrunc(draw_pos + cursor_offset - draw_scroll);
            ImRect cursor_screen_rect(cursor_screen_pos.x, cursor_screen_pos.y - g.FontSize + 0.5f, cursor_screen_pos.x + 1.0f, cursor_screen_pos.y - 1.5f);
            if (cursor_is_visible && cursor_screen_rect.Overlaps(clip_rect))
                draw_window->DrawList->AddLine(cursor_screen_rect.Min, cursor_screen_rect.GetBL(), GetColorU32(ImGuiCol_Text));

            // Notify OS of text input position for advanced IME (-1 x offset so that Windows IME can cover our cursor. Bit of an extra nicety.)
            if (!is_readonly)
            {
                g.PlatformImeData.WantVisible = true;
                g.PlatformImeData.InputPos = ImVec2(cursor_screen_pos.x - 1.0f, cursor_screen_pos.y - g.FontSize);
                g.PlatformImeData.InputLineHeight = g.FontSize;
                g.PlatformImeViewport = window->Viewport->ID;
            }
        }
    }
    else
    {
        // Render text only (no selection, no cursor)
        if (is_multiline)
            text_size = ImVec2(inner_size.x, InputTextCalcTextLenAndLineCount(buf_display, &buf_display_end) * line_size); // We don't need width
        else if (!is_displaying_hint && g.ActiveId == id)
            buf_display_end = buf_display + state->TextLen;
        else if (!is_displaying_hint)
            buf_display_end = buf_display + strlen(buf_display);

        if (is_multiline || (buf_display_end - buf_display) < buf_display_max_length)
        {
            // Find render position for right alignment
            if (flags & ImGuiInputTextFlags_ElideLeft)
                draw_pos.x = ImMin(draw_pos.x, frame_bb.Max.x - CalcTextSize(buf_display, NULL).x - style.FramePadding.x);

            const ImVec2 draw_scroll = /*state ? ImVec2(state->Scroll.x, 0.0f) :*/ ImVec2(0.0f, 0.0f); // Preserve scroll when inactive?
            ImU32 col = GetColorU32(is_displaying_hint ? ImGuiCol_TextDisabled : ImGuiCol_Text);
            
			int offs = 0;
			for (const char* s = buf_display, *snext; s != NULL && s < buf_display_end; s = snext)
			{
				snext = (const char*)memchr(s, '\n', (size_t)(buf_display_end - s));
				
				draw_window->DrawList->AddText(g.Font, g.FontSize, draw_pos - draw_scroll + ImVec2(0, g.FontSize + offs * line_size), col, s, snext, 0.0f, is_multiline ? NULL : &clip_rect);
				
				if(snext!=NULL) snext++; // advance past \n
				
				++offs;
			}
        }
    }

    if (is_password && !is_displaying_hint)
        PopFont();

    if (is_multiline)
    {
        // For focus requests to work on our multiline we need to ensure our child ItemAdd() call specifies the ImGuiItemFlags_Inputable (see #4761, #7870)...
        Dummy(ImVec2(text_size.x, text_size.y + style.FramePadding.y));
        g.NextItemData.ItemFlags |= (ImGuiItemFlags)ImGuiItemFlags_Inputable | ImGuiItemFlags_NoTabStop;
        EndChild();
        item_data_backup.StatusFlags |= (g.LastItemData.StatusFlags & ImGuiItemStatusFlags_HoveredWindow);

        // ...and then we need to undo the group overriding last item data, which gets a bit messy as EndGroup() tries to forward scrollbar being active...
        // FIXME: This quite messy/tricky, should attempt to get rid of the child window.
        EndGroup();
        if (g.LastItemData.ID == 0 || g.LastItemData.ID != GetWindowScrollbarID(draw_window, ImGuiAxis_Y))
        {
            g.LastItemData.ID = id;
            g.LastItemData.ItemFlags = item_data_backup.ItemFlags;
            g.LastItemData.StatusFlags = item_data_backup.StatusFlags;
        }
    }
    if (state)
        state->TextSrc = NULL;

    // Log as text
    if (g.LogEnabled && (!is_password || is_displaying_hint))
    {
        LogSetNextTextDecoration("{", "}");
        LogRenderedText(&draw_pos, buf_display, buf_display_end);
    }

    if (label_size.x > 0)
        RenderText(ImVec2(frame_bb.Max.x + style.ItemInnerSpacing.x, frame_bb.Min.y + style.FramePadding.y), label);

    if (value_changed)
        MarkItemEdited(id);

    IMGUI_TEST_ENGINE_ITEM_INFO(id, label, g.LastItemData.StatusFlags | ImGuiItemStatusFlags_Inputable);
    if ((flags & ImGuiInputTextFlags_EnterReturnsTrue) != 0)
        return validated;
    else
        return value_changed;
}