# ACMDM Motion Generation

Generate human motion animations from text descriptions using the ACMDM (Absolute Coordinates Make Motion Generation Easy) model.

## 🎯 Features

- **Text-to-Motion Generation**: Create realistic human motion from natural language descriptions
- **Batch Processing**: Generate multiple motions at once
- **Auto-Length Estimation**: AI automatically determines optimal motion length
- **Flexible Parameters**: Adjust CFG scale, motion length, and more
- **Real-time Preview**: See your generated motions instantly

## Results

<img width="581" height="650" alt="image" src="https://github.com/user-attachments/assets/af3e8503-b417-481d-b94b-2322a998c632" />
<img width="581" height="650" alt="image" src="https://github.com/user-attachments/assets/42ef7545-e3be-4715-856b-f1c4ee84ab60" />
<img width="581" height="650" alt="image" src="https://github.com/user-attachments/assets/7bd0f379-9300-4bf7-9530-800e5f768364" />

## 🚀 Usage

1. **Enter a text description** of the motion you want (e.g., "A person is running on a treadmill.")
2. **Adjust parameters** (optional):
   - Motion length (40-196 frames)
   - CFG scale (controls text alignment)
   - Auto-length estimation
3. **Click "Generate Motion"**
4. **View and download** your generated motion video

## 📝 Example Prompts

- "A person is running on a treadmill."
- "Someone is doing jumping jacks."
- "A person walks forward and then turns around."
- "A person is dancing energetically."

## ⚙️ Parameters

- **Motion Length**: Number of frames (40-196). Automatically rounded to multiples of 4.
- **CFG Scale**: Classifier-free guidance scale (1.0-10.0). Higher = more text-aligned, Lower = more diverse.
- **Auto-length**: Let AI estimate the optimal motion length based on your text.

## 🔧 Technical Details

This space uses pre-trained ACMDM models:
- **Autoencoder**: AE_2D_Causal
- **Diffusion Model**: ACMDM_Flow_S_PatchSize22
- **Dataset**: HumanML3D (t2m)

## 📚 Paper

[Absolute Coordinates Make Motion Generation Easy](https://arxiv.org/abs/2505.19377)

## 🤝 Citation

```bibtex
@article{meng2025absolute,
    title={Absolute Coordinates Make Motion Generation Easy},
    author={Meng, Zichong and Han, Zeyu and Peng, Xiaogang and Xie, Yiming and Jiang, Huaizu},
    journal={arXiv preprint arXiv:2505.19377},
    year={2025}
}
```

## ⚠️ Notes

- First generation may take 30-60 seconds (model loading)
- Subsequent generations are faster (5-15 seconds)
- GPU recommended for best performance
- Works on CPU but slower

---
title: ACMDM Motion Generation
emoji: 🎭
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
hardware: gpu-t4-small
---


## 🔗 Links

- [GitHub Repository](https://github.com/neu-vi/ACMDM)
- [Project Page](https://neu-vi.github.io/ACMDM/)
- [Paper](https://arxiv.org/abs/2505.19377)

