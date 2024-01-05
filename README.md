# Voice Cloning & Avatar Generator


[![Python 3.10.13](https://img.shields.io/badge/python-3.10.13-blue.svg)](https://www.python.org/downloads/release/python-31013/)
[![PyTorch 2.1.2](https://img.shields.io/badge/pytorch-2.1.2-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit 1.29.0](https://img.shields.io/badge/streamlit-1.29.0-FF4B4B.svg)](https://streamlit.io/)
[![Docker Support](https://img.shields.io/badge/docker-support-2496ED.svg)](https://www.docker.com/)
![Built with Love](https://img.shields.io/badge/built%20with-%E2%9D%A4-red.svg)


Welcome to the Voice Cloning & Avatar Generator project! This innovative project leverages two powerful models, SadTalker and XTTS, to deliver a unique and captivating experience for voice cloning across various languages, including Arabic, English, Spanish, French, German, Italian, and many others. Additionally, it offers cutting-edge 3D avatar generation.

## Features

- **Multilingual Support:** Our voice cloning feature supports a wide range of languages, providing a versatile experience for users globally.
- **SadTalker Model:** Empowered by the SadTalker model, our application not only replicates various emotional tones but also excels in lip sync, ensuring a seamless and realistic experience.
- **XTTS Model:** The XTTS model enhances the quality and naturalness of generated voices, guaranteeing an immersive user experience.
- **Advanced Animation:** With the SadTalker model, you can take advantage of advanced animation features. Capture head movements, eye blinks, and other facial expressions from a provided video, adding an extra layer of realism to your avatars.
- **Extended Generation Time:** Achieving exceptional quality takes time. The application allows for longer generation times to ensure the highest fidelity in both voice and avatar.
- Docker support for easy deployment.

## Getting Started

To get started with the Voice Cloning & Avatar Generator, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/alifallaha1/talking-avatar-and-voice-cloning.git
   ```

2. Install the required dependencies. We recommend using a virtual environment:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch the web application:
   
   ```bash
   streamlit run app.py
   ```

4. Access the web application by opening `http://localhost:8501` in your browser.
#####(hint):when you run it for the first time it will take time to download the cheakpoint for xtts 
## Docker Support

The Voice Cloning & Avatar Generator also provides Docker support for easy deployment. To build and run the Docker image, follow these steps:

1. Build the Docker image:

   ```bash
   docker build -t avatar_clone .
   ```
  
2. Run the Docker container:
   
   ```bash
   docker run --rm --runtime=nvidia --gpus all avtar_clone

   ```
   
4. Access the web application by opening `http://localhost:8501` in your browser.

## Contributing

Contributions are welcome and greatly appreciated! To contribute to the Voice Cloning & Avatar Generator project, follow these steps:

1. Fork the repository.

2. Create a new branch:

   ```bash
   git checkout -b feature/my-feature
   ```

3. Make the desired changes and commit them:
   
   ```bash
   git commit -m "Add my feature"
   ```

4. Push to the branch:
      
   ```bash
   git push origin feature/my-feature
   ```

5. Open a pull request in the main repository.


## Contact

If you have any questions, suggestions, or feedback, please feel free to contact me:

- GitHub: [https://github.com/alifallaha1](https://github.com/alifallaha1)
- LinkedIn : [https://linkedin.com/in/ali-wael-](https://linkedin.com/in/ali-wael-)

I'm open to collaboration and look forward to hearing from you!

---

Thank you for visiting the PRNU Predictor repository. I hope you find it useful and informative. Happy device identification using PRNU values!

