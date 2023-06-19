# SPDX-FileCopyrightText: 2023-present dh031200 <imbird0312@gmail.com>
#
# SPDX-License-Identifier: Apache-2.0
import gradio as gr


def app(inference):
    demo = gr.Interface(
        inference,
        inputs=[gr.Image(label="Image"),
                gr.Textbox(label="Prompt-class (ex: black cat:cat, white dog:dog, ...)"),
                gr.Slider(0, 1, 0.25, label="Box Threshold"),
                gr.Slider(0, 1, 0.8, label="NMS Threshold"),
                ],
        outputs=[gr.Image(label="Detection"),
                 gr.Image(label="Segmentation"),
                 gr.Image(label="Mask"),
                 gr.Json(label="json"),
                 ],
    )
    demo.launch(share=False, server_name="0.0.0.0")
