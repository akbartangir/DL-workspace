# **Setup Instructions: Install Miniconda**

For this project, we use the Conda package manager to manage dependencies. Follow the instructions below to install Miniconda for your operating system.

---

## **Windows Installation**
1. Open a command prompt (`cmd`).
2. Run the following commands to quickly and silently install Miniconda:
   ```cmd
   curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe -o miniconda.exe
   start /wait "" .\miniconda.exe /S
   del miniconda.exe
    ```
3. Close the command prompt.

---

## **MacOS Installation**
1. Open a terminal.
2. Run the following commands to quickly and silently install Miniconda:
   ```bash
   mkdir -p ~/miniconda3
   curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -o ~/miniconda3/miniconda.sh
   bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
   rm ~/miniconda3/miniconda.sh
   ```
3. After installing, close and reopen your terminal application or refresh it by running the following command:
   ```bash
   source ~/miniconda3/bin/activate
   conda init --all
   ```

---

## **Linux Installation**
1. Open a terminal.
2. Run the following commands to quickly and silently install Miniconda:
   ```bash
   mkdir -p ~/miniconda3
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
   bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
   rm ~/miniconda3/miniconda.sh
   ```
3. After installing, close and reopen your terminal application or refresh it by running the following command:
   ```bash
    source ~/miniconda3/bin/activate
    conda init --all
    ```

