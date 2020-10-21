mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"mail@mail.dk\"n\
" > ~/.streamlit/credentials.toml

echo "[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml