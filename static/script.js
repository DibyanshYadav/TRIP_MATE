/* ------------------------------------------------------------------
   TripMate AI front end: chat style
   - keeps the whole conversation visible
   - clears the input after sending
   - keeps chat + thread id + map pins across refreshes (this tab only: a fresh open starts a new chat)
   - streams live progress (SSE) while the agents work
   - draws the trip map (Leaflet + OpenStreetMap)
------------------------------------------------------------------- */

const STORE_THREAD = "travel_thread_id";
const STORE_CHAT = "travel_chat_history";

// one colour per day, repeats after 10 days
const DAY_COLORS = [
    "#ef4444", "#f59e0b", "#22c55e", "#06b6d4", "#3b82f6",
    "#8b5cf6", "#ec4899", "#14b8a6", "#f97316", "#84cc16"
];

let currentThreadId = sessionStorage.getItem(STORE_THREAD) || null;
let chatHistory = loadHistory();      // [{ role: "user" | "agent", text, pins? }]
let isBusy = false;
let lastUserText = "";

let chatEpoch = 0;     // bumps on "New chat" so late map pins never land in the wrong chat
let reqSeq = 0;        // id of the latest request
let activeReq = null;  // request that currently holds the "busy" state

/* ---------------------------- helpers ---------------------------- */

const $ = (id) => document.getElementById(id);

function loadHistory() {
    try {
        const raw = sessionStorage.getItem(STORE_CHAT);
        const parsed = raw ? JSON.parse(raw) : [];
        return Array.isArray(parsed) ? parsed : [];
    } catch (e) {
        return [];
    }
}

function saveHistory() {
    try {
        sessionStorage.setItem(STORE_CHAT, JSON.stringify(chatHistory));
    } catch (e) { /* storage full or blocked: ignore */ }
}

function renderMarkdown(text) {
    if (typeof marked === "undefined") return null;
    let html = marked.parse(text);
    if (typeof DOMPurify !== "undefined") html = DOMPurify.sanitize(html);
    return html;
}

function updateThreadInfo() {
    $("threadInfo").textContent = currentThreadId
        ? `Thread ${String(currentThreadId).slice(0, 8)}`
        : "New conversation";
}

function updateWelcome() {
    $("welcome").classList.toggle("hidden", chatHistory.length > 0);
}

function scrollToBottom(smooth = true) {
    const chat = $("chat");
    chat.scrollTo({ top: chat.scrollHeight, behavior: smooth ? "smooth" : "auto" });
}

function autoGrow() {
    const input = $("userInput");
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 200) + "px";
}

/* --------------------------- UI state ---------------------------- */

function setLoading(loading) {
    isBusy = loading;
    $("sendBtn").disabled = loading;
    $("btnText").classList.toggle("hidden", loading);
    $("btnLoader").classList.toggle("hidden", !loading);
}

function showError(message) {
    const box = $("errorBox");
    box.textContent = message;
    box.classList.remove("hidden");
}

function hideError() {
    const box = $("errorBox");
    box.classList.add("hidden");
    box.textContent = "";
}

/* ------------------------------ map ------------------------------ */

function renderMap(rowEl, pins) {
    if (typeof L === "undefined" || !Array.isArray(pins)) return;

    const valid = pins.filter((p) => isFinite(Number(p.lat)) && isFinite(Number(p.lon)));
    if (valid.length === 0) return;

    const bubble = rowEl.querySelector(".bubble");
    if (!bubble) return;

    // re-rendering (e.g. pins arrive twice): remove the old map first
    bubble.querySelectorAll(".trip-map, .map-note").forEach((n) => n.remove());

    const mapEl = document.createElement("div");
    mapEl.className = "trip-map";

    const note = document.createElement("div");
    note.className = "map-note";
    note.textContent = "Pins are approximate (OpenStreetMap search). The number is the day.";

    // keep the Copy / PDF buttons at the bottom of the bubble
    const actions = bubble.querySelector(".msg-actions");
    bubble.insertBefore(mapEl, actions);
    bubble.insertBefore(note, actions);

    const map = L.map(mapEl, { scrollWheelZoom: false });
    L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
        maxZoom: 19,
        attribution: "&copy; OpenStreetMap contributors"
    }).addTo(map);

    // wheel-zoom only after the user clicks the map, so chat scrolling is not hijacked
    map.on("click", () => map.scrollWheelZoom.enable());
    mapEl.addEventListener("mouseleave", () => map.scrollWheelZoom.disable());

    const bounds = [];
    const byDay = {};

    valid.forEach((pin) => {
        const lat = Number(pin.lat);
        const lon = Number(pin.lon);
        const day = Number(pin.day) || 1;
        const color = DAY_COLORS[(day - 1 + DAY_COLORS.length) % DAY_COLORS.length];

        // only numbers and a constant colour go into this HTML string, so it is safe
        const icon = L.divIcon({
            className: "",
            html: `<div class="day-pin" style="background:${color}">${day}</div>`,
            iconSize: [28, 28],
            iconAnchor: [14, 14],
            popupAnchor: [0, -14]
        });

        // popup content is a DOM node with textContent: the place name is never parsed as HTML
        const popup = document.createElement("div");
        popup.textContent = `Day ${day}: ${pin.name}`;

        L.marker([lat, lon], { icon }).addTo(map).bindPopup(popup);

        bounds.push([lat, lon]);
        (byDay[day] = byDay[day] || []).push([lat, lon]);
    });

    // dashed line connecting each day's stops in order
    Object.entries(byDay).forEach(([day, pts]) => {
        if (pts.length < 2) return;
        const color = DAY_COLORS[(Number(day) - 1 + DAY_COLORS.length) % DAY_COLORS.length];
        L.polyline(pts, { color, weight: 3, opacity: 0.7, dashArray: "6 6" }).addTo(map);
    });

    map.fitBounds(bounds, { padding: [30, 30], maxZoom: 14 });

    // the container was just added to the page, so let Leaflet re-measure it
    setTimeout(() => map.invalidateSize(), 150);
}

function showMapNote(rowEl, text) {
    const bubble = rowEl && rowEl.querySelector(".bubble");
    if (!bubble) return;
    bubble.querySelectorAll(".map-note").forEach((n) => n.remove());

    const note = document.createElement("div");
    note.className = "map-note";
    note.textContent = text;
    bubble.insertBefore(note, bubble.querySelector(".msg-actions"));
}

function attachPins(el, pins, epoch) {
    if (epoch !== chatEpoch || !el || !el.isConnected) return;

    renderMap(el, pins);

    // remember the pins so the map comes back after a refresh
    const idx = Number(el.dataset.idx);
    if (Number.isInteger(idx) && chatHistory[idx]) {
        chatHistory[idx].pins = pins;
        saveHistory();
    }
}

/* ------------------------- message rendering ---------------------- */

function buildMessageEl(role, text) {
    const row = document.createElement("div");
    row.className = `msg ${role}`;

    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.textContent = role === "user" ? "🧑" : "✈️";

    const bubble = document.createElement("div");
    bubble.className = "bubble";

    const body = document.createElement("div");
    body.className = "md";

    if (role === "agent") {
        const html = renderMarkdown(text);
        if (html !== null) {
            body.innerHTML = html;
            body.querySelectorAll("a").forEach((a) => {
                a.target = "_blank";
                a.rel = "noopener noreferrer";
            });
        } else {
            body.textContent = text;
        }
    } else {
        body.textContent = text;   // user text is never treated as HTML
    }

    bubble.appendChild(body);

    if (role === "agent") {
        const actions = document.createElement("div");
        actions.className = "msg-actions";

        const copyBtn = document.createElement("button");
        copyBtn.textContent = "Copy";
        copyBtn.onclick = () => copyText(text, copyBtn);

        const pdfBtn = document.createElement("button");
        pdfBtn.textContent = "PDF";
        pdfBtn.onclick = () => downloadPDF(body, pdfBtn);

        actions.append(copyBtn, pdfBtn);
        bubble.appendChild(actions);
    }

    row.append(avatar, bubble);
    return row;
}

function addMessage(role, text, { save = true, scroll = true, pins = null } = {}) {
    const el = buildMessageEl(role, text);
    $("messages").appendChild(el);

    if (save) {
        chatHistory.push({ role, text });
        saveHistory();
        el.dataset.idx = String(chatHistory.length - 1);
    }

    // the map needs the element to be in the page first (Leaflet measures it)
    if (pins && pins.length) renderMap(el, pins);

    updateWelcome();

    if (scroll) {
        if (role === "agent") {
            // long answers: show the START of the reply, not the end
            el.scrollIntoView({ behavior: "smooth", block: "start" });
        } else {
            scrollToBottom();
        }
    }
    return el;
}

function showTyping() {
    const row = document.createElement("div");
    row.className = "msg agent";
    row.id = "typingRow";
    row.innerHTML = `
        <div class="avatar">✈️</div>
        <div class="bubble typing">
            <div class="typing-line">
                <span></span><span></span><span></span>
                <em id="typingText">Understanding your request…</em>
            </div>
            <ul class="steps" id="stepsList"></ul>
        </div>`;
    $("messages").appendChild(row);
    scrollToBottom();
}

function updateStep(step) {
    const list = $("stepsList");
    if (!list || !step || !step.name) return;

    const text = $("typingText");
    if (text) text.textContent = "Working on it… planning can take a minute";

    // one line per tool; a second call of the same tool just updates its line
    let li = null;
    list.querySelectorAll("li").forEach((n) => {
        if (n.dataset.name === step.name) li = n;
    });

    if (!li) {
        li = document.createElement("li");
        li.dataset.name = step.name;

        const icon = document.createElement("span");
        icon.className = "step-icon";

        const label = document.createElement("span");
        label.className = "step-label";

        li.append(icon, label);
        list.appendChild(li);
    }

    li.className = `step ${step.state === "done" ? "done" : "running"}`;
    li.querySelector(".step-label").textContent = step.label || step.name;
    scrollToBottom();
}

function hideTyping() {
    const row = $("typingRow");
    if (row) row.remove();
}

function showErrorBubble(message) {
    const row = document.createElement("div");
    row.className = "msg agent error-msg";
    row.id = "errorRow";

    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.textContent = "⚠️";

    const bubble = document.createElement("div");
    bubble.className = "bubble";

    const text = document.createElement("div");
    text.textContent = message;

    const retry = document.createElement("button");
    retry.className = "retry-btn";
    retry.textContent = "↻ Retry";
    retry.onclick = () => {
        row.remove();
        sendMessage(lastUserText, { skipUserBubble: true });
    };

    bubble.append(text, retry);
    row.append(avatar, bubble);
    $("messages").appendChild(row);
    scrollToBottom();
}

/* ---------------------------- streaming --------------------------- */

/**
 * POST to the streaming endpoint and call onEvent(eventName, data) for every
 * Server-Sent Event. (EventSource can only do GET, so we parse the stream by hand.)
 */
async function streamTravel(message, onEvent) {
    const response = await fetch("/api/travel/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            messages: message,            // backend expects "messages"
            thread_id: currentThreadId
        })
    });

    // validation errors (400 / 422) come back as normal JSON, not as a stream
    if (!response.ok || !response.body) {
        let data = {};
        try {
            data = await response.json();
        } catch (e) {
            throw new Error(`Server error (${response.status}). Check the terminal for details.`);
        }
        const detail = data.detail ? JSON.stringify(data.detail) : null;
        throw new Error(data.error || detail || "Something went wrong.");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    const dispatch = (block) => {
        let event = "message";
        const dataLines = [];

        block.split("\n").forEach((line) => {
            if (line.startsWith("event:")) event = line.slice(6).trim();
            else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
        });

        if (dataLines.length === 0) return;

        let payload;
        try {
            payload = JSON.parse(dataLines.join("\n"));
        } catch (e) {
            return;   // ignore a broken event instead of killing the whole stream
        }
        onEvent(event, payload);
    };

    while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");

        // events are separated by a blank line
        let sep;
        while ((sep = buffer.indexOf("\n\n")) !== -1) {
            dispatch(buffer.slice(0, sep));
            buffer = buffer.slice(sep + 2);
        }
    }

    if (buffer.trim()) dispatch(buffer);
}

/* ---------------------------- actions ---------------------------- */

function setPrompt(text) {
    const input = $("userInput");
    input.value = text;
    autoGrow();
    input.focus();
}

async function sendMessage(overrideText, { skipUserBubble = false } = {}) {
    if (isBusy) return;

    hideError();

    const input = $("userInput");
    const message = (typeof overrideText === "string" ? overrideText : input.value).trim();

    if (!message) {
        showError("Please type a message first.");
        return;
    }

    lastUserText = message;

    // show the user's message and clear the box immediately
    if (!skipUserBubble) {
        addMessage("user", message);
    }
    input.value = "";
    autoGrow();
    input.focus();

    const myReq = ++reqSeq;
    activeReq = myReq;
    const epoch = chatEpoch;

    setLoading(true);
    showTyping();

    let answered = false;
    let agentEl = null;

    try {
        await streamTravel(message, (event, data) => {
            if (event === "step") {
                updateStep(data);

            } else if (event === "answer") {
                answered = true;

                currentThreadId = data.thread_id;
                sessionStorage.setItem(STORE_THREAD, currentThreadId);
                updateThreadInfo();

                hideTyping();
                if (epoch === chatEpoch) {
                    agentEl = addMessage("agent", data.answer);
                }

                // the text is here, so unlock the input now. Map pins are still being
                // looked up in the background and will appear on this message later.
                if (activeReq === myReq) {
                    activeReq = null;
                    setLoading(false);
                }

            } else if (event === "pins") {
                console.log("[TripMate] pins event:", data);
                if (agentEl && epoch === chatEpoch) {
                    if (data.pins && data.pins.length) {
                        if (typeof L === "undefined") {
                            showMapNote(agentEl, "Map library (Leaflet) did not load. Check your internet / ad blocker.");
                        } else {
                            attachPins(agentEl, data.pins, epoch);
                        }
                    } else if (data.places_found) {
                        showMapNote(agentEl, "Map unavailable: places were found but could not be located. Check the server terminal for 'Geocode failed'.");
                    } else {
                        showMapNote(agentEl, "Map unavailable: this reply had no itinerary places (no PLACES_JSON line).");
                    }
                }

            } else if (event === "error") {
                throw new Error(data.error || "Something went wrong.");
            }
        });

        if (!answered) {
            throw new Error("No response received from the server.");
        }

    } catch (error) {
        if (answered) {
            // the answer is already on screen; only the map step failed
            console.warn("Stream failed after the answer was sent:", error);
        } else {
            hideTyping();
            showErrorBubble(error.message);
        }
    } finally {
        // only release the busy state if a newer request has not taken it over
        if (activeReq === myReq) {
            activeReq = null;
            setLoading(false);
        }
    }
}

function newChat() {
    if (isBusy) return;

    if (chatHistory.length > 0 && !confirm("Start a new chat? The current conversation will be cleared.")) {
        return;
    }

    chatEpoch++;
    currentThreadId = null;
    chatHistory = [];
    lastUserText = "";
    sessionStorage.removeItem(STORE_THREAD);
    sessionStorage.removeItem(STORE_CHAT);

    $("messages").innerHTML = "";
    hideError();
    updateThreadInfo();
    updateWelcome();
    $("userInput").focus();
}

function copyText(text, btn) {
    navigator.clipboard.writeText(text)
        .then(() => {
            const old = btn.textContent;
            btn.textContent = "Copied!";
            setTimeout(() => (btn.textContent = old), 1400);
        })
        .catch(() => showError("Could not copy."));
}

function downloadPDF(bodyEl, btn) {
    const old = btn.textContent;
    btn.textContent = "Preparing…";
    btn.disabled = true;

    // render into a white, print-style sheet (the chat itself is dark)
    const sheet = document.createElement("div");
    sheet.className = "pdf-sheet";
    sheet.innerHTML = `<h1 class="pdf-title">AI Travel Plan</h1>` + bodyEl.innerHTML;
    document.body.appendChild(sheet);

    const options = {
        margin: 0.5,
        filename: "ai-travel-plan.pdf",
        image: { type: "jpeg", quality: 0.98 },
        html2canvas: { scale: 2, useCORS: true, backgroundColor: "#ffffff" },
        jsPDF: { unit: "in", format: "a4", orientation: "portrait" },
        pagebreak: { mode: ["avoid-all", "css", "legacy"] }
    };

    const done = () => {
        sheet.remove();
        btn.textContent = old;
        btn.disabled = false;
    };

    html2pdf().set(options).from(sheet).save()
        .then(done)
        .catch(() => {
            done();
            showError("Could not download PDF.");
        });
}

/* ------------------------------ init ----------------------------- */

document.addEventListener("DOMContentLoaded", () => {
    const input = $("userInput");

    // Enter sends, Shift+Enter makes a new line
    input.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
            e.preventDefault();
            sendMessage();
        }
    });
    input.addEventListener("input", autoGrow);

    // restore previous conversation (with its maps)
    chatHistory.forEach((m) =>
        addMessage(m.role, m.text, { save: false, scroll: false, pins: m.pins || null })
    );
    updateThreadInfo();
    updateWelcome();
    scrollToBottom(false);
    input.focus();
});