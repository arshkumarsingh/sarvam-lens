# Source this before each `modal` command:  source modal_env.example.sh
# KEYFILE is a text file containing your `modal token set --token-id ak-... --token-secret as-...`
# line. The token is exported for this shell only, never echoed, and never written to ~/.modal.toml.
KEYFILE="${KEYFILE:-$HOME/modal_token.txt}"
export MODAL_TOKEN_ID="$(grep -oE 'ak-[A-Za-z0-9]+' "$KEYFILE" | head -1)"
export MODAL_TOKEN_SECRET="$(grep -oE 'as-[A-Za-z0-9]+' "$KEYFILE" | head -1)"
export PYTHONIOENCODING=utf-8
