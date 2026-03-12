import os
import gradio as gr
import huggingface_hub
from transformers import pipeline
import re
import jiwer
import Levenshtein
from datetime import datetime
import tempfile
from fpdf import FPDF

# Use the normalize function from our existing utils.py
from utils import normalize

# Import our new feedback DB and Patient DB
from rag.feedback_db import save_feedback, get_all_feedback
from rag.vectorstore import add_document
from database import init_db, load_mock_data, get_all_patients, get_patient_scans, get_scan_by_accession

# Initialize Patient DB and load mock data
init_db()
load_mock_data()

# ---------------------------------------------------------------------------
# Model loading (lazy for Whisper, eager for MedASR)
# ---------------------------------------------------------------------------
MODEL_ID = "google/medasr"
print("Loading MedASR model into pipeline... (this may take a moment)")
pipe = pipeline("automatic-speech-recognition", model=MODEL_ID)
print("Model loaded successfully!")

_whisper_pipe = None

def _get_whisper_pipe():
    """Lazy-load Whisper for Turkish / auto-detect transcription."""
    global _whisper_pipe
    if _whisper_pipe is None:
        import torch
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        print(f"Loading Whisper model on {device}...")
        _whisper_pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-large-v3",
            device=device,
            torch_dtype=dtype,
        )
        print("Whisper model loaded.")
    return _whisper_pipe

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_html_diff(ref_text: str, hyp_text: str):
    """Calculates WER and builds a beautiful HTML visual diff."""
    normalized_ref = normalize(ref_text)
    normalized_hyp = normalize(hyp_text)

    ref_words = normalized_ref.split()
    hyp_words = normalized_hyp.split()

    # Calculate metrics
    measures = jiwer.process_words([normalized_ref], [normalized_hyp])
    edits = Levenshtein.editops(ref_words, hyp_words)

    r = 0
    diff = []

    for op, i, j in edits:
        if r < i:
            for word in ref_words[r:i]:
                diff.append(f'<span style="background-color:rgba(34, 197, 94, 0.2); padding:2px; border-radius:3px; margin: 0 2px;">{word}</span>')
        r = i

        if op == 'replace':
            diff.append(f'<del style="color:#b91c1c; background-color:#fef2f2; text-decoration:line-through; padding:2px; border-radius:3px; margin: 0 2px;">{ref_words[i]}</del>')
            diff.append(f'<ins style="color:#15803d; background-color:#f0fdf4; text-decoration:none; padding:2px; border-radius:3px; font-weight:bold; margin: 0 2px;">{hyp_words[j]}</ins>')
            r += 1
        elif op == 'insert':
            diff.append(f'<ins style="color:#15803d; background-color:#f0fdf4; text-decoration:none; padding:2px; border-radius:3px; font-weight:bold; margin: 0 2px;">{hyp_words[j]}</ins>')
        elif op == 'delete':
            diff.append(f'<del style="color:#b91c1c; background-color:#fef2f2; text-decoration:line-through; padding:2px; border-radius:3px; margin: 0 2px;">{ref_words[i]}</del>')
            r += 1

    if r < len(ref_words):
        for word in ref_words[r:]:
            diff.append(f'<span style="background-color:rgba(34, 197, 94, 0.2); padding:2px; border-radius:3px; margin: 0 2px;">{word}</span>')

    wer_str = (
        f'<div style="font-size: 1.1em; margin-bottom: 10px;">'
        f'<b>Word Error Rate (WER): {measures.wer * 100:.2f}%</b>'
        f'</div>'
        f'<div style="font-size: 0.9em; color: #666; margin-bottom: 15px;">'
        f'Insertions: {measures.insertions} | Deletions: {measures.deletions} | '
        f'Substitutions: {measures.substitutions} | Ref tokens: {len(ref_words)}'
        f'</div>'
    )
    diff_html = f'<div style="line-height: 1.8; font-size: 1.05em; padding: 15px; background-color: var(--block-background-fill); color: var(--body-text-color); border-radius: 8px; border: 1px solid var(--border-color-primary);">{" ".join(diff)}</div>'
    
    return wer_str + diff_html


def simple_html_diff(old_text: str, new_text: str):
    """A simpler HTML diff for comparing two strings (e.g. original vs edited report)."""
    old_words = old_text.split()
    new_words = new_text.split()
    
    edits = Levenshtein.editops(old_words, new_words)
    r = 0
    diff = []

    for op, i, j in edits:
        if r < i:
            for word in old_words[r:i]:
                diff.append(f'<span>{word}</span>')
        r = i

        if op == 'replace':
            diff.append(f'<del style="color:#b91c1c; text-decoration:line-through;">{old_words[i]}</del>')
            diff.append(f'<ins style="color:#15803d; text-decoration:none; font-weight:bold;">{new_words[j]}</ins>')
            r += 1
        elif op == 'insert':
            diff.append(f'<ins style="color:#15803d; text-decoration:none; font-weight:bold;">{new_words[j]}</ins>')
        elif op == 'delete':
            diff.append(f'<del style="color:#b91c1c; text-decoration:line-through;">{old_words[i]}</del>')
            r += 1

    if r < len(old_words):
        for word in old_words[r:]:
            diff.append(f'<span>{word}</span>')
            
    return f'<div style="line-height: 1.6; padding: 10px; border: 1px solid #ddd; border-radius: 5px;">{" ".join(diff)}</div>'

# ---------------------------------------------------------------------------
# Transcription handler (supports language selection)
# ---------------------------------------------------------------------------

def transcribe(audio_path, ref_transcript="", language="Auto"):
    if not audio_path:
        return "Please provide an audio input.", "", gr.update(visible=False, value="")
    
    try:
        detected_lang = "en"

        if language == "English":
            result = pipe(audio_path, chunk_length_s=20, stride_length_s=2)
            transcribed_text = result["text"]
            detected_lang = "en"
        elif language == "Turkish":
            wpipe = _get_whisper_pipe()
            result = wpipe(
                audio_path,
                chunk_length_s=20,
                stride_length_s=2,
                generate_kwargs={"task": "transcribe", "language": "turkish"},
            )
            transcribed_text = result["text"]
            detected_lang = "tr"
        else:  # Auto
            wpipe = _get_whisper_pipe()
            result = wpipe(audio_path, chunk_length_s=20, stride_length_s=2)
            text = result.get("text", "")
            turkish_chars = set("çğıöşüÇĞİÖŞÜ")
            if any(c in text for c in turkish_chars):
                transcribed_text = text
                detected_lang = "tr"
            else:
                # Re-transcribe with MedASR for better English medical accuracy
                result = pipe(audio_path, chunk_length_s=20, stride_length_s=2)
                transcribed_text = result["text"]
                detected_lang = "en"

        lang_label = "🇹🇷 Turkish" if detected_lang == "tr" else "🇬🇧 English"
        
        # Calculate evaluation diff if reference transcript provided
        if ref_transcript and ref_transcript.strip():
            eval_html = get_html_diff(ref_transcript, transcribed_text)
            return transcribed_text, lang_label, gr.update(visible=True, value=eval_html)
        else:
            return transcribed_text, lang_label, gr.update(visible=False, value="")
            
    except Exception as e:
        return f"Error: {str(e)}", "", gr.update(visible=False, value="")

# ---------------------------------------------------------------------------
# Turkish report generation handler
# ---------------------------------------------------------------------------

def generate_report(transcript_text, llm_backend_choice):
    """Generate a Turkish medical report from the transcription using RAG."""
    if not transcript_text or not transcript_text.strip():
        return "⚠️ Önce ses dosyasını transkribe edin. (Transcribe audio first.)", ""
    
    try:
        from rag.report_generator import generate_turkish_report

        backend_map = {
            "LM Studio (Local)": "lmstudio",
            "OpenAI (GPT-4o-mini)": "openai",
            "Ollama (Local)": "ollama",
            "HuggingFace (Local)": "huggingface",
        }
        backend = backend_map.get(llm_backend_choice, "openai")

        report = generate_turkish_report(
            transcript=transcript_text,
            language="auto",
            llm_backend=backend,
        )
        # Returns (current_report, original_report_state)
        return report, report
    except Exception as e:
        return f"❌ Rapor oluşturulurken hata: {str(e)}", ""

# ---------------------------------------------------------------------------
# Full Pipeline: Transcribe + Generate Report
# ---------------------------------------------------------------------------
def transcribe_and_generate(audio_path, ref_transcript, language, llm_backend_choice):
    """Runs the full pipeline: Audio -> Text -> Turkish Medical Report."""
    if not audio_path:
        return "Please provide an audio input.", "", gr.update(visible=False, value=""), "", ""
    
    # 1. Transcribe
    transcribed_text, lang_label, eval_html = transcribe(audio_path, ref_transcript, language)
    
    if transcribed_text.startswith("Error:"):
        return transcribed_text, lang_label, eval_html, "⚠️ Transkripsiyon hatası olduğu için rapor oluşturulamadı.", ""

    # 2. Generate Report
    report, _ = generate_report(transcribed_text, llm_backend_choice)
    
    return transcribed_text, lang_label, eval_html, report, report


# ---------------------------------------------------------------------------
# Feedback & Learning Mechanism
# ---------------------------------------------------------------------------
def handle_save_feedback(transcript, original_report, edited_report):
    """Save user edits to DB and push to vector store."""
    if not original_report or not edited_report:
        return "⚠️ Kaydedilecek rapor bulunamadı."
        
    if original_report.strip() == edited_report.strip():
        return "ℹ️ Orijinal rapor üzerinde herhangi bir değişiklik yapmadınız."
    
    # 1. Save to SQLite
    save_feedback(transcript, original_report, edited_report)
    
    # 2. Add the corrected version back into ChromaDB Vector Store for future RAG queries
    metadata = {
        "source": "user_feedback",
        "type": "corrected_report",
        "transcript_context": transcript[:100]  # store a snippet of the transcript
    }
    # We prefix it so the LLM knows this is a golden example
    vector_doc = f"Örnek Doğru Rapor:\n{edited_report}"
    add_document(vector_doc, metadata)
    
    return "✅ Değişiklikleriniz başarıyla kaydedildi ve bilgi tabanına (Vector Store) eklendi! Gelecekteki raporlar bu düzeltmelerden öğrenecek."

def load_feedback_history():
    """Load feedback history from DB and return as formatted HTML list."""
    history = get_all_feedback()
    if not history:
        return "Henüz kaydedilmiş bir düzeltme bulunmuyor."
        
    html = "<h3>Geçmiş Düzeltmeler (Eğitim Verisi)</h3><div style='display: flex; flex-direction: column; gap: 15px;'>"
    for item in history:
        diff_html = simple_html_diff(item["original_report"], item["edited_report"])
        html += f"""
        <div style="border: 1px solid var(--border-color-primary); border-radius: 8px; padding: 15px;">
            <p><strong>Tarih:</strong> {item['timestamp'][:16].replace('T', ' ')}</p>
            <p><strong>İlgili Transkript:</strong> <i>{item['transcript']}</i></p>
            <details>
                <summary style="cursor: pointer; font-weight: bold; margin-bottom: 10px;">Farklılıkları (Diff) Gör</summary>
                <div style="margin-top: 10px;">{diff_html}</div>
            </details>
        </div>
        """
    html += "</div>"
    return html

# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

# The built-in sample transcript
SAMPLE_TRANSCRIPT = (
    "Exam type CT chest PE protocol period. Indication 54 year old female, "
    "shortness of breath, evaluate for PE period. Technique standard protocol period. "
    "Findings colon. Pulmonary vasculature colon. The main PA is patent period. "
    "There are filling defects in the segmental branches of the right lower lobe comma "
    "compatible with acute PE period. No saddle embolus period. Lungs colon. "
    "No pneumothorax period. Small bilateral effusions comma right greater than left period. "
    "New paragraph. Impression colon Acute segmental PE right lower lobe period."
)

# Try fetching sample audio to populate the examples
try:
    sample_audio = huggingface_hub.hf_hub_download(MODEL_ID, "test_audio.wav")
    has_sample = True
except Exception:
    has_sample = False

# ---------------------------------------------------------------------------
# Patient and Report Handlers
# ---------------------------------------------------------------------------

def get_patient_choices():
    patients = get_all_patients()
    # Format: "Patient Name (ID)"
    return [f"{p['full_name']} ({p['patient_id']})" for p in patients]

def update_scan_choices(patient_selection):
    if not patient_selection:
        return gr.update(choices=[], value=None)
    
    try:
        # Extract ID from "Name (ID)"
        patient_id = patient_selection.split("(")[-1].strip(")")
        scans = get_patient_scans(patient_id)
        
        if not scans:
            return gr.update(choices=[], value=None)
            
        # Format: "Date - Clinic (Accession: ACC-XXX)"
        choices = [f"{s.scan_date} - {s.clinic} (Accession: {s.accession_number})" for s in scans]
        return gr.update(choices=choices, value=choices[0] if choices else None)
    except Exception as e:
        print(f"Error updating scans: {e}")
        return gr.update(choices=[], value=None)

def update_patient_info(scan_selection):
    if not scan_selection:
        return "Lütfen bir çekim seçiniz."
        
    try:
        # Extract accession number
        accession = scan_selection.split("Accession: ")[-1].strip(")")
        scan = get_scan_by_accession(accession)
        
        if not scan:
            return "Çekim bilgisi bulunamadı."
            
        info = f"""
        **Hasta Adı Soyadı:** {scan.full_name}
        **Protokol Numarası:** {scan.protocol_number}
        **Cinsiyet:** {scan.gender}
        **Doğum Tarihi:** {scan.dob}
        **Çekim Tarihi:** {scan.scan_date}
        **Erişim Numarası (Accession):** {scan.accession_number}
        **Klinik:** {scan.clinic}
        **LOINC Kodu:** {scan.loinc_code}
        """
        return info
    except Exception as e:
        return f"Bilgi getirilirken hata: {e}"

def get_report_title_by_loinc(loinc_code):
    mapping = {
        "24606-6": "Tarama Mamografisi Raporu",
        "24604-1": "Tanısal Mamografi Raporu",
        "26041-4": "Meme Ultrasonografi Raporu"
    }
    return mapping.get(loinc_code, "Radyoloji Raporu")

def generate_final_report(scan_selection, generated_body):
    if not scan_selection or not generated_body:
        return "Lütfen hasta/çekim seçin ve rapor metnini oluşturun."
        
    try:
        accession = scan_selection.split("Accession: ")[-1].strip(")")
        scan = get_scan_by_accession(accession)
        
        if not scan:
            return "Çekim bilgisi bulunamadı."
            
        title = get_report_title_by_loinc(scan.loinc_code)
        
        final_report = f"""{title}
        
HASTA BİLGİLERİ (HEADER)
----------------------------------------
Adı Soyadı      : {scan.full_name}
Protokol No     : {scan.protocol_number}
Erişim No       : {scan.accession_number}
Doğum Tarihi    : {scan.dob}
Çekim Tarihi    : {scan.scan_date}
Klinik          : {scan.clinic}
----------------------------------------

KLİNİK BULGULAR VE SONUÇ (BODY)
----------------------------------------
{generated_body}
"""
        return final_report
    except Exception as e:
        return f"Rapor birleştirilirken hata: {e}"

def create_pdf(final_report_text):
    if not final_report_text or "Lütfen hasta/çekim seçin" in final_report_text:
        return gr.update(value=None, visible=True)
        
    # Create PDF
    pdf = FPDF()
    pdf.add_page()
    
    # Use standard Arial/Helvetica but fallback replace Turkish chars to avoid errors
    pdf.set_font("Helvetica", size=12)
    
    # We replace common Turkish characters that might cause encoding issues in basic fonts
    replacements = {
        'ğ': 'g', 'Ğ': 'G', 
        'ş': 's', 'Ş': 'S', 
        'ı': 'i', 'İ': 'I',
        'ö': 'o', 'Ö': 'O',
        'ç': 'c', 'Ç': 'C',
        'ü': 'u', 'Ü': 'U'
    }
    
    safe_text = final_report_text
    for tr, eng in replacements.items():
        safe_text = safe_text.replace(tr, eng)
        
    pdf.multi_cell(0, 10, txt=safe_text)
        
    # Save to a temporary file and return the path wrapped in gr.update
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(temp_file.name)
    
    return gr.update(value=temp_file.name, visible=True)

# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="MedASR + Turkish Report Generator") as demo:
    
    # We use a state to track the ORIGINAL unedited report 
    # so we can compare it with the USER EDITED version.
    original_report_state = gr.State("")
    
    gr.Markdown(
        """
        # 🩺 MedASR – Medical ASR & Turkish Report Generator
        Record your voice or upload audio to transcribe medical terminology, then generate a professional Turkish medical report using RAG.
        """
    )
    
    with gr.Tabs():
        # --- TAB 1: Main Pipeline ---
        with gr.Tab("Rapor Oluştur"):
            with gr.Row():
                # ---- Left column: inputs ----
                with gr.Column(scale=1):
                    # Patient Selection Section
                    gr.Markdown("### 1. Hasta Seçimi")
                    with gr.Group():
                        patient_selector = gr.Dropdown(
                            choices=get_patient_choices(),
                            label="Hasta Seçin",
                            info="Sisteme kayıtlı hastalar (İsim (ID))"
                        )
                        scan_selector = gr.Dropdown(
                            choices=[],
                            label="Çekim (Accession) Seçin",
                            info="Hastaya ait çekimler"
                        )
                        patient_info_display = gr.Markdown("Lütfen bir çekim seçiniz.")

                    gr.Markdown("---")
                    gr.Markdown("### 2. Ses ve Transkripsiyon")
                    audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Audio Input")
                    
                    lang_selector = gr.Radio(
                        choices=["Auto", "English", "Turkish"],
                        value="Auto",
                        label="Audio Language",
                        info="Auto-detect or choose the language of the audio",
                    )

                    llm_backend_selector = gr.Dropdown(
                        choices=["LM Studio (Local)", "OpenAI (GPT-4o-mini)", "Ollama (Local)", "HuggingFace (Local)"],
                        value="LM Studio (Local)",
                        label="LLM Backend",
                        info="Select the LLM to use for report generation.",
                    )
                    
                    with gr.Accordion("Evaluate (Optional)", open=False):
                        gr.Markdown("Provide a reference transcript to calculate Word Error Rate (WER) and see a visual diff.")
                        ref_input = gr.Textbox(
                            lines=4, 
                            placeholder="Enter reference transcript here...", 
                            label="Reference Transcript",
                            value=SAMPLE_TRANSCRIPT if has_sample else ""
                        )
                    
                    # Action buttons
                    with gr.Row():
                        submit_btn = gr.Button("🎙️ Sadece Transkribe Et", size="lg")
                        report_btn = gr.Button("📝 Sadece Rapor Üret", size="lg")
                    
                    full_pipeline_btn = gr.Button("🚀 Transkribe Et & Rapor Oluştur (Tek Tık)", variant="primary", size="lg")
                    
                # ---- Right column: outputs ----
                with gr.Column(scale=1):
                    text_output = gr.Textbox(label="Transkripsiyon (Raw Text)", lines=5)
                    detected_lang_output = gr.Textbox(label="Detected Language", lines=1, interactive=False)
                    eval_output = gr.HTML(label="Evaluation (WER & Diff)", visible=False)
                    
                    gr.Markdown("---")
                    gr.Markdown("### 3. Üretilen Tıbbi Bulgu Metni (Body)")
                    gr.Markdown("*Aşağıdaki klinik bulgu metnini (Body) düzenleyebilirsiniz.*")
                    report_output = gr.Textbox(
                        label="Klinik Bulgular ve Sonuç (Body)",
                        lines=10,
                        interactive=True, # Critical: Allow user to edit the output
                    )
                    
                    save_feedback_btn = gr.Button("💾 Düzeltmeleri Kaydet ve Sisteme Öğret", variant="primary")
                    feedback_status = gr.Textbox(label="Kayıt Durumu", interactive=False, lines=1)

                    gr.Markdown("---")
                    gr.Markdown("### 4. Nihai Birleştirilmiş Rapor")
                    generate_final_btn = gr.Button("📄 Nihai Raporu Oluştur (Header + Body)", variant="secondary")
                    final_report_output = gr.Textbox(label="Nihai Rapor", lines=15, interactive=False)
                    
                    pdf_download_btn = gr.DownloadButton("📥 PDF Olarak İndir")

            # ---- Wire up events for TAB 1 ----
            # 0. Patient Selection Events
            patient_selector.change(
                fn=update_scan_choices,
                inputs=[patient_selector],
                outputs=[scan_selector]
            )
            
            scan_selector.change(
                fn=update_patient_info,
                inputs=[scan_selector],
                outputs=[patient_info_display]
            )

            # 1. Only Transcribe
            submit_btn.click(
                fn=transcribe, 
                inputs=[audio_input, ref_input, lang_selector], 
                outputs=[text_output, detected_lang_output, eval_output]
            )
            
            # 2. Only Generate Report
            report_btn.click(
                fn=generate_report,
                inputs=[text_output, llm_backend_selector],
                outputs=[report_output, original_report_state], # save original to state
            )

            # 3. Full Pipeline
            full_pipeline_btn.click(
                fn=transcribe_and_generate,
                inputs=[audio_input, ref_input, lang_selector, llm_backend_selector],
                outputs=[text_output, detected_lang_output, eval_output, report_output, original_report_state],
            )
            
            # 4. Save feedback
            save_feedback_btn.click(
                fn=handle_save_feedback,
                inputs=[text_output, original_report_state, report_output],
                outputs=[feedback_status]
            )
            
            # 5. Generate Final Report & PDF
            generate_final_btn.click(
                fn=generate_final_report,
                inputs=[scan_selector, report_output],
                outputs=[final_report_output]
            )
            
            final_report_output.change(
                fn=create_pdf,
                inputs=[final_report_output],
                outputs=[pdf_download_btn]
            )

            if has_sample:
                gr.Examples(
                    examples=[[sample_audio, SAMPLE_TRANSCRIPT, "Auto"]],
                    inputs=[audio_input, ref_input, lang_selector],
                    outputs=[text_output, detected_lang_output, eval_output],
                    fn=transcribe,
                    cache_examples=False,
                    label="Try it out with sample audio"
                )

        # --- TAB 2: Learning History & Diffs ---
        with gr.Tab("Veritabanı ve Düzeltme Geçmişi"):
            gr.Markdown("Bu sekmede kullanıcılar tarafından düzeltilmiş ve RAG bilgi tabanına eklenmiş raporların orijinalleriyle karşılaştırmalarını (diff) görebilirsiniz. Sistem yeni oluşturacağı raporlarda buradaki öğrenilmiş örnekleri dikkate alacaktır.")
            
            refresh_btn = gr.Button("🔄 Geçmişi Yenile")
            history_output = gr.HTML()
            
            refresh_btn.click(
                fn=load_feedback_history,
                inputs=[],
                outputs=[history_output]
            )
            
            # Load on initial tab switch/load
            demo.load(
                fn=load_feedback_history,
                inputs=[],
                outputs=[history_output]
            )

if __name__ == "__main__":
    allowed = [os.path.dirname(sample_audio)] if has_sample else None
    demo.launch(
        server_name="127.0.0.1", 
        server_port=7860, 
        share=False, 
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="indigo"),
        allowed_paths=allowed
    )
