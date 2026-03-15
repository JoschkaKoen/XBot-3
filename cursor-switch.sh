#!/usr/bin/env bash
# cursor-switch.sh — Switch Cursor IDE between model provider profiles
#
# Cursor stores:
#   - API key:     in the OS secret store (gnome-keyring on Linux)
#   - Base URL:    in state.vscdb (SQLite), inside a JSON blob
#   - Toggle states: same JSON blob in state.vscdb
#
# Usage:
#   ./cursor-switch.sh discover         # find keyring entries & DB keys
#   ./cursor-switch.sh list             # list configured profiles
#   ./cursor-switch.sh use <profile>    # switch to a profile
#   ./cursor-switch.sh off              # disable BYOK, use Cursor defaults
#
# Profiles are stored in ~/.config/cursor-switch/profiles.conf
#
# ──────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────
CURSOR_DATA="${XDG_CONFIG_HOME:-$HOME/.config}/Cursor"
STATE_DB="$CURSOR_DATA/User/globalStorage/state.vscdb"
STATE_DB_BACKUP="$STATE_DB.backup"
CONF_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/cursor-switch"
PROFILES_FILE="$CONF_DIR/profiles.conf"

# The key in ItemTable that holds the big JSON blob with all Cursor settings
REACTIVE_KEY="src.vs.platform.reactivestorage.browser.reactiveStorageServiceImpl.persistentStorage.applicationUser"

# ── Colors ────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# ── Helpers ───────────────────────────────────────────────────────────

die() { echo -e "${RED}ERROR:${NC} $*" >&2; exit 1; }
info() { echo -e "${CYAN}→${NC} $*"; }
ok()   { echo -e "${GREEN}✓${NC} $*"; }
warn() { echo -e "${YELLOW}⚠${NC} $*"; }

require_cmd() {
    command -v "$1" &>/dev/null || die "'$1' is required but not found. Install it first."
}

check_cursor_running() {
    if pgrep -f "[Cc]ursor" &>/dev/null; then
        warn "Cursor appears to be running."
        warn "Changes will take effect after restarting Cursor."
        echo ""
    fi
}

ensure_db() {
    [[ -f "$STATE_DB" ]] || die "Cursor state DB not found at: $STATE_DB"
}

backup_db() {
    cp "$STATE_DB" "$STATE_DB.switch-backup-$(date +%s)"
    ok "Database backed up"
}

ensure_conf() {
    mkdir -p "$CONF_DIR"
    if [[ ! -f "$PROFILES_FILE" ]]; then
        cat > "$PROFILES_FILE" <<'EXAMPLE'
# Cursor Switch — Provider Profiles
#
# Format:  PROFILE_NAME|API_KEY|BASE_URL
#
# PROFILE_NAME : short name you'll use on the command line
# API_KEY      : your API key for this provider
# BASE_URL     : the OpenAI-compatible base URL (with /v1 etc.)
#
# Examples:
# zai|glm-xxxxxxxxxxxxxxxx|https://api.z.ai/api/coding/paas/v4
# minimax|your-minimax-key-here|https://api.minimax.chat/v1
# openrouter|sk-or-v1-xxxx|https://openrouter.ai/api/v1
# scaleway|scw-secret-key|https://api.scaleway.ai/v1
#
# Lines starting with # are comments.
EXAMPLE
        warn "Created $PROFILES_FILE — edit it to add your profiles."
        return 1
    fi
    return 0
}

load_profile() {
    local name="$1"
    local line
    line=$(grep -v '^\s*#' "$PROFILES_FILE" | grep -v '^\s*$' | grep "^${name}|" | head -1)
    if [[ -z "$line" ]]; then
        die "Profile '$name' not found in $PROFILES_FILE"
    fi
    PROFILE_NAME=$(echo "$line" | cut -d'|' -f1)
    PROFILE_KEY=$(echo "$line" | cut -d'|' -f2)
    PROFILE_URL=$(echo "$line" | cut -d'|' -f3)
}

# ── DB operations ─────────────────────────────────────────────────────

# Read a field from the reactive storage JSON
db_read_field() {
    local field="$1"
    sqlite3 "$STATE_DB" \
        "SELECT json_extract(value, '\$.${field}')
         FROM ItemTable
         WHERE key = '${REACTIVE_KEY}';" 2>/dev/null || echo ""
}

# Write a field in the reactive storage JSON
db_write_field() {
    local field="$1" val="$2"
    sqlite3 "$STATE_DB" \
        "UPDATE ItemTable
         SET value = json_set(value, '\$.${field}', ${val})
         WHERE key = '${REACTIVE_KEY}';"
}

# ── Keyring operations ────────────────────────────────────────────────
# Cursor (Electron) typically stores secrets via libsecret / gnome-keyring.
# The exact attributes depend on your Cursor version. The discover command
# helps you find them.

KEYRING_SERVICE=""
KEYRING_ACCOUNT=""

discover_keyring() {
    info "Searching gnome-keyring for Cursor-related entries..."
    echo ""

    # Method 1: search via secret-tool (if available)
    if command -v secret-tool &>/dev/null; then
        info "Dumping keyring items matching 'cursor' or 'openai'..."
        echo ""

        # Use seahorse/secret-tool to search — unfortunately secret-tool
        # doesn't support wildcard search, so we try known patterns.

        local found=0
        for svc in "cursor" "cursor.ai" "cursor-editor" "Cursor" \
                    "vscode" "code" "codium"; do
            local result
            result=$(secret-tool search service "$svc" 2>/dev/null || true)
            if [[ -n "$result" ]]; then
                echo -e "${GREEN}Found entries for service='$svc':${NC}"
                echo "$result" | head -30
                echo ""
                found=1
            fi
        done

        if (( found == 0 )); then
            warn "No entries found via secret-tool search."
            info "Trying alternative: searching with 'xdg-schema'..."
            for schema in "org.gnome.keyring.Note" "org.freedesktop.Secret.Generic"; do
                result=$(secret-tool search xdg:schema "$schema" 2>/dev/null | head -40 || true)
                if [[ -n "$result" ]]; then
                    echo -e "${GREEN}Found entries for schema='$schema':${NC}"
                    echo "$result"
                    echo ""
                fi
            done
        fi
    fi

    # Method 2: check if Cursor uses file-based fallback
    local fallback_dir="$CURSOR_DATA/User/globalStorage/state-secrets"
    if [[ -d "$fallback_dir" ]]; then
        echo -e "${GREEN}Found file-based secret storage:${NC} $fallback_dir"
        ls -la "$fallback_dir" 2>/dev/null || true
        echo ""
    fi

    # Method 3: check for Electron safeStorage
    local safe_storage="$CURSOR_DATA/Local State"
    if [[ -f "$safe_storage" ]]; then
        info "Found Electron Local State (may contain encrypted key storage info):"
        echo "  $safe_storage"
        echo ""
    fi

    # Method 4: search the SQLite DB for any key-related entries
    if [[ -f "$STATE_DB" ]]; then
        info "Searching state.vscdb for API-key-related keys..."
        echo ""
        sqlite3 "$STATE_DB" \
            "SELECT key FROM ItemTable
             WHERE key LIKE '%openai%' OR key LIKE '%apiKey%'
                OR key LIKE '%api_key%' OR key LIKE '%baseUrl%'
                OR key LIKE '%base_url%' OR key LIKE '%override%';" 2>/dev/null | while read -r k; do
            echo -e "  ${CYAN}$k${NC}"
        done
        echo ""

        info "Reading current model settings from reactive storage..."
        echo ""

        # Try common field paths in the reactive JSON blob
        for field in \
            "aiService.openaiApiKey" \
            "aiService.useOpenaiApiKey" \
            "aiService.openaiBaseUrl" \
            "aiService.useOpenaiBaseUrl" \
            "aiService.overrideOpenaiBaseUrl" \
            "modelService.openaiApiKey" \
            "modelService.useOpenaiApiKey" \
            "modelService.openaiBaseUrl" \
            "modelService.useOpenaiBaseUrl" \
            "modelService.overrideOpenaiBaseUrl" \
            "cursorSettings.openaiApiKey" \
            "cursorSettings.useOpenaiApiKey" \
            "cursorSettings.openaiBaseUrl" \
            "cursorSettings.overrideOpenaiBaseUrl" \
        ; do
            local val
            val=$(db_read_field "$field")
            if [[ -n "$val" && "$val" != "null" ]]; then
                # Mask API keys
                local display_val="$val"
                if [[ "$field" == *"apiKey"* || "$field" == *"api_key"* ]]; then
                    if (( ${#val} > 8 )); then
                        display_val="${val:0:4}...${val: -4}"
                    fi
                fi
                echo -e "  ${GREEN}$field${NC} = ${BOLD}$display_val${NC}"
            fi
        done
        echo ""

        # Dump ALL top-level keys in the reactive JSON so the user can find the right ones
        info "Top-level keys in reactive storage JSON:"
        sqlite3 "$STATE_DB" \
            "SELECT key, json_each.key AS jkey
             FROM ItemTable, json_each(ItemTable.value)
             WHERE ItemTable.key = '${REACTIVE_KEY}';" 2>/dev/null \
        | grep -iE 'openai|apikey|api_key|baseurl|base_url|override|model' \
        | while read -r line; do
            echo -e "  ${CYAN}$line${NC}"
        done
        echo ""

        # More aggressive: dump the entire JSON and grep
        info "Grepping full reactive JSON for openai/apiKey/baseUrl patterns..."
        local full_json
        full_json=$(sqlite3 "$STATE_DB" \
            "SELECT value FROM ItemTable WHERE key = '${REACTIVE_KEY}';" 2>/dev/null || echo "")
        if [[ -n "$full_json" ]]; then
            echo "$full_json" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    def walk(obj, path=''):
        if isinstance(obj, dict):
            for k, v in obj.items():
                full = f'{path}.{k}' if path else k
                if any(x in k.lower() for x in ['openai', 'apikey', 'api_key', 'baseurl', 'base_url', 'override']):
                    display = v
                    if isinstance(v, str) and ('key' in k.lower() or 'api' in k.lower()) and len(v) > 8:
                        display = v[:4] + '...' + v[-4:]
                    print(f'  \033[0;32m{full}\033[0m = \033[1m{display}\033[0m')
                if isinstance(v, (dict, list)):
                    walk(v, full)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                walk(item, f'{path}[{i}]')
    walk(data)
except Exception as e:
    print(f'  (parse error: {e})', file=sys.stderr)
" 2>/dev/null || warn "Could not parse reactive JSON"
        fi
    fi

    echo ""
    echo -e "${BOLD}═══ What to do next ═══${NC}"
    echo ""
    echo "1. Look at the output above to find the exact JSON field paths for:"
    echo "   - The OpenAI API key enable toggle"
    echo "   - The base URL override toggle"
    echo "   - The base URL value"
    echo ""
    echo "2. Edit the FIELD PATHS section at the top of this script if they"
    echo "   differ from the defaults."
    echo ""
    echo "3. If the API key is in gnome-keyring (not in the DB), note the"
    echo "   service/account attributes and update the KEYRING_* variables."
    echo ""
    echo "4. Add your profiles to: $PROFILES_FILE"
    echo ""
}

# ── Field paths (update these after running 'discover') ───────────────
# These are the most likely paths based on community research.
# Run './cursor-switch.sh discover' to confirm for your Cursor version.

# Toggle: enable the OpenAI API key
FIELD_USE_KEY="aiService.useOpenaiApiKey"
# Toggle: enable the base URL override
FIELD_USE_BASE_URL="aiService.useOpenaiBaseUrl"
# The actual base URL value
FIELD_BASE_URL="aiService.openaiBaseUrl"
# The API key (if stored in DB rather than keyring)
FIELD_API_KEY="aiService.openaiApiKey"

# ── Commands ──────────────────────────────────────────────────────────

cmd_discover() {
    require_cmd sqlite3
    ensure_db
    discover_keyring
}

cmd_list() {
    ensure_conf || exit 0
    echo -e "${BOLD}Configured profiles:${NC}"
    echo ""
    grep -v '^\s*#' "$PROFILES_FILE" | grep -v '^\s*$' | while IFS='|' read -r name key url; do
        local masked_key="$key"
        if (( ${#key} > 8 )); then
            masked_key="${key:0:4}...${key: -4}"
        fi
        echo -e "  ${GREEN}$name${NC}"
        echo -e "    Key: ${masked_key}"
        echo -e "    URL: ${url}"
    done
    echo ""

    # Show current state
    ensure_db
    local cur_url cur_key_enabled cur_url_enabled
    cur_url=$(db_read_field "$FIELD_BASE_URL")
    cur_key_enabled=$(db_read_field "$FIELD_USE_KEY")
    cur_url_enabled=$(db_read_field "$FIELD_USE_BASE_URL")

    echo -e "${BOLD}Current state:${NC}"
    echo -e "  BYOK enabled:     ${cur_key_enabled:-unknown}"
    echo -e "  URL override:     ${cur_url_enabled:-unknown}"
    echo -e "  Base URL:         ${cur_url:-<not set>}"
}

cmd_use() {
    local profile_name="$1"
    require_cmd sqlite3
    ensure_db
    ensure_conf || die "No profiles configured. Edit $PROFILES_FILE first."
    load_profile "$profile_name"

    check_cursor_running
    backup_db

    info "Switching to profile: ${BOLD}${PROFILE_NAME}${NC}"

    # 1. Enable the OpenAI API key toggle
    info "Enabling OpenAI API key override..."
    db_write_field "$FIELD_USE_KEY" "true"

    # 2. Set the API key (if it's stored in the DB)
    info "Setting API key..."
    db_write_field "$FIELD_API_KEY" "\"${PROFILE_KEY}\""

    # 3. Enable the base URL override
    info "Enabling base URL override..."
    db_write_field "$FIELD_USE_BASE_URL" "true"

    # 4. Set the base URL
    info "Setting base URL to: ${PROFILE_URL}"
    db_write_field "$FIELD_BASE_URL" "\"${PROFILE_URL}\""

    # 5. Also try storing the key in gnome-keyring if secret-tool is available
    if command -v secret-tool &>/dev/null && [[ -n "$KEYRING_SERVICE" ]]; then
        info "Updating gnome-keyring..."
        echo -n "$PROFILE_KEY" | secret-tool store \
            --label="Cursor OpenAI API Key" \
            service "$KEYRING_SERVICE" \
            account "$KEYRING_ACCOUNT" 2>/dev/null || \
            warn "Could not update keyring (non-fatal)"
    fi

    echo ""
    ok "Switched to ${BOLD}${PROFILE_NAME}${NC}"
    warn "Restart Cursor for changes to take effect."
    echo ""
}

cmd_off() {
    require_cmd sqlite3
    ensure_db
    check_cursor_running
    backup_db

    info "Disabling BYOK — reverting to Cursor default models..."

    # Disable the toggles
    db_write_field "$FIELD_USE_KEY" "false"
    db_write_field "$FIELD_USE_BASE_URL" "false"

    echo ""
    ok "BYOK disabled. Cursor will use its default subscription models."
    warn "Restart Cursor for changes to take effect."
    echo ""
}

# ── Main ──────────────────────────────────────────────────────────────

usage() {
    echo -e "${BOLD}cursor-switch${NC} — Switch Cursor IDE model provider profiles"
    echo ""
    echo "Usage:"
    echo "  $0 discover           Find where Cursor stores keys/URLs on your system"
    echo "  $0 list               List configured profiles + current state"
    echo "  $0 use <profile>      Switch to a profile (sets key + base URL)"
    echo "  $0 off                Disable BYOK, revert to Cursor defaults"
    echo ""
    echo "Config: $PROFILES_FILE"
    echo ""
}

case "${1:-}" in
    discover)  cmd_discover ;;
    list)      cmd_list ;;
    use)
        [[ -n "${2:-}" ]] || die "Usage: $0 use <profile_name>"
        cmd_use "$2"
        ;;
    off)       cmd_off ;;
    *)         usage ;;
esac