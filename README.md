Installation:
```console
git clone https://github.com/RuinedOustrich/Rag-AI-Helper.git
cd Rag-AI-Helper
pip install -r requirements.txt
```
Download model here: https://drive.google.com/file/d/1CMNHutAYUz7y5JKQW8W22TiBQRUIIUSE/view?usp=sharing
and put it in /home/user/Rag-AI-Helper/model

You can run the streamlit app by:
```console
./run.sh path_to_project
```
To run app in shell:
```console
python main.py path_to_project
```
path_to_project is None by default. That means that you wil get additional contexts only from external database(if you define it by changing database_path in config) and from web.
