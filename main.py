import webview
import gradio as gr
import whisper

model = whisper.load_model("base")

title = "Speech Recognition"
description = "Transcribe long-form microphone or audio inputs with a single click!"


def transcribe_mic(audio, task):
    transcribed = model.transcribe(audio, task=task)
    result = transcribed["text"]

    return result


def transcribe_au(audio, task):
    transcribed = model.transcribe(audio, task=task)
    result = transcribed["text"]

    return result


mf_transcribe = gr.Interface(
    fn=transcribe_mic,
    inputs=[
        gr.Audio(source="microphone", type="filepath"),
        gr.Radio(["transcribe", "translate"], label="Task"),
    ],
    outputs="text",
    title="Whisper: Transcribe Audio",
    description=description,
    allow_flagging="never",
)

au_transcribe = gr.Interface(
    fn=transcribe_au,
    inputs=[
        gr.Audio(source="upload", type="filepath"),
        gr.Radio(["transcribe", "translate"], label="Task"),
    ],
    outputs="text",
    title="Whisper: Transcribe Audio",
    description=description,
    allow_flagging="never",
)

demo = gr.TabbedInterface([mf_transcribe, au_transcribe], ["Transcribe Microphone",
                                                           "Transcribe Audio",
                                                           ],
                          css="footer {visibility: hidden}"
                          )


def launch_webview():
    window = webview.create_window(
                                    "Whisper",
                                    "http://127.0.0.1:7860/",
                                    confirm_close=True,
                                    width=1000,
                                    height=800,
                                    min_size=(400, 200)
                                    )
    window.events.closed += demo.close
    webview.start(private_mode=True, storage_path=None)


def launch_gradio():
    demo.launch(prevent_thread_lock=True)


launch_gradio()
launch_webview()
