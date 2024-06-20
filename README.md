```sh
python src/analyze.py path/to/directory "Describe the image"
```

<p align="center">
  <img src="assets/image.png" alt="Consume-Blip3 Logo" width="500"/>
</p>

# BLIP3 Autocaptioning Tools

Welcome to this XGEN-MM(BLIP3) Autocaptioning Tools repository! This project sets up tools for autocaptioning using state-of-the-art models. 

<div align="center">
<p>
✅ Chat Mode &emsp;&emsp;&emsp;
✅ Caption Mode &emsp;&emsp;&emsp;
✅ FastAPI Application 
</p>
</div>


## Table of Contents

- [Introduction](#introduction)
- [Setup](#setup)
- [Usage](docs/USAGE.md)
- [Contributing](#contributing)
- [License](#license)

## Introduction

XGEN (BLIP3) is designed to provide efficient and accurate (lol to a degree) autocaptioning capabilities. This repository aims to set up the necessary environment and some tools to leverage the power of the XGen-MM-PHI3 model from Salesforce.

## Setup
(TESTED ON UBUNTU 22.04 | CUDA 12.1 | Torch 2.3.1+cu121) <br>
*For windows, lmk. I'll make a pull request to actually test. but should work fine.*
<br><br>
Follow the steps below to set up the project:

### Option 1: Using Setup Scripts

#### Linux/Mac:
1. **Download and Run the Shell Script:**
    ```sh
    wget https://raw.githubusercontent.com/C0nsumption/Consume-Blip3/main/setup/setup.sh
    chmod +x setup.sh
    ./setup.sh
    ```

#### Windows:
1. **Download and Run the Batch Script:**
   ```bat
    curl -o setup.bat https://raw.githubusercontent.com/C0nsumption/Consume-Blip3/main/setup/setup.bat
    setup.bat
   ```
### Option 2: Manual Installation
<details>
  <summary>Manual Installation</summary>

1. **Clone this Repo and Navigate to the Project Directory:**
    ```sh
    git clone https://github.com/C0nsumption/Consume-Blip3.git
    cd Consume-Blip3
    ```

2. **Set Up a Virtual Environment:**
    ```sh
    python -m venv venv
    source venv/bin/activate  # For Linux/Mac
    venv\Scripts\activate  # For Windows
    ```

3. **Initialize with Git LFS (make sure to have installed. Ask ChatGPT.):**
    ```sh
    git lfs install
    ```

4. **Clone the Model Repository:**
    ```sh
    git clone https://huggingface.co/Salesforce/xgen-mm-phi3-mini-instruct-r-v1
    ```

5. **Install Dependencies:**
    ```sh
    pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
    pip install -r requirements.txt
    ```

6. **Run Tests:**
    ```sh
    python test/test.py
    ```

</details>

## Usage

After setting up the environment, you can start using the BLIP3 autocaptioning tools. Detailed usage instructions and examples can be found in the [Usage Guide](docs/USAGE.md).

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
