<p align="center">
  <img src="image.png" alt="Consume-Blip3 Logo" width="500"/>
</p>

# BLIP3 Autocaptioning Tools

Welcome to this XGEN-MM(BLIP3) Autocaptioning Tools repository! This project sets up tools for autocaptioning using state-of-the-art models. 

## Table of Contents

- [Introduction](#introduction)
- [Setup](#setup)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

XGEN (BLIP3) is designed to provide efficient and accurate (lol to a degree) autocaptioning capabilities. This repository aims to set up the necessary environment and some tools to leverage the power of the XGen-MM-PHI3 model from Salesforce.

## Setup
(TESTED ON UBUNTU 22.04 | CUDA 12.1 | Torch 2.3.1+cu121) <br>
*For windows, lmk. I'll make a pull request.*
<br><br>
Follow the steps below to set up the project:

1. **Create and Navigate to Project Directory:**
    ```sh
    mkdir blip3 && cd blip3
    ```

2. **Install Git LFS:**
    ```sh
    git lfs install
    ```

3. **Clone the Model Repository:**
    ```sh
    git clone https://huggingface.co/Salesforce/xgen-mm-phi3-mini-instruct-r-v1
    ```

4. **Install Dependencies:**
    ```sh
    pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
    pip install -r requirements.txt
    ```

5. **Run Tests:**
    ```sh
    python test.py
    ```

## Usage

After setting up the environment, you can start using the BLIP3 autocaptioning tools. Detailed usage instructions and examples will be provided in a `docs` folder at some point.

## Contributing

I welcome contributions from the community! If you'd like to contribute, please fork the repository and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Feel free to reach out if you have any questions or need further assistance! But give me time, very busy: 
<br>
ａｃｃｅｌｅｒａｔｉｎｇ 🫡

---