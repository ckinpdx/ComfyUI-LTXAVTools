import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

// Portrait use-case labels keyed by landscape ratio string
const PORTRAIT_LABELS = {
    "16:9": "9:16 — TikTok, Reels, Shorts",
    "21:9": "9:21 — Ultrawide portrait",
    "4:3":  "3:4 — Tablet portrait",
    "3:2":  "2:3 — Portrait photo",
    "2:1":  "1:2 — Tall mobile",
    "5:4":  "4:5 — Instagram portrait",
    "1:1":  "1:1 — Square, Instagram",
};

const LANDSCAPE_LABELS = {
    "16:9": "16:9 — YouTube, HD, TV",
    "21:9": "21:9 — Ultrawide, cinematic",
    "4:3":  "4:3 — Classic TV, monitor",
    "3:2":  "3:2 — Photography, DSLR",
    "2:1":  "2:1 — Cinematic wide",
    "5:4":  "5:4 — Old CRT monitor",
    "1:1":  "1:1 — Square, Instagram",
};

// Extract the bare ratio key from either label format
// "16:9 — YouTube, HD, TV"  →  "16:9"
// "9:16 — TikTok, Reels, Shorts"  →  "9:16"  →  normalise to landscape  →  "16:9"
function toRatioKey(label) {
    const raw = label.split("—")[0].trim();   // "16:9" or "9:16"
    const [a, b] = raw.split(":").map(Number);
    // Always return as landscape (wider:taller)
    return a >= b ? `${a}:${b}` : `${b}:${a}`;
}

app.registerExtension({
    name: "ltx.DimensionCalculator",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "LTXDimensionCalculator") return;

        const onAdded = nodeType.prototype.onAdded;
        nodeType.prototype.onAdded = function () {
            onAdded?.apply(this, arguments);

            const ratioWidget       = this.widgets.find(w => w.name === "ratio");
            const orientationWidget = this.widgets.find(w => w.name === "orientation");
            const resolutionWidget  = this.widgets.find(w => w.name === "resolution");

            if (!ratioWidget || !orientationWidget || !resolutionWidget) return;

            // ------------------------------------------------------------------
            // Fetch updated resolution list from Python API and refresh widget
            // ------------------------------------------------------------------
            const fetchResolutions = async () => {
                const ratioKey    = toRatioKey(ratioWidget.value);
                const orientation = orientationWidget.value;

                try {
                    const resp = await api.fetchApi(
                        `/ltx-dim-calc/resolutions` +
                        `?ratio=${encodeURIComponent(ratioKey)}` +
                        `&orientation=${encodeURIComponent(orientation)}`
                    );
                    const options = await resp.json();
                    if (!options?.length) return;

                    const prev = resolutionWidget.value;
                    resolutionWidget.options.values = options;
                    resolutionWidget.value = options.includes(prev)
                        ? prev
                        : options[Math.floor(options.length / 2)];

                    app.graph.setDirtyCanvas(true, true);
                } catch (err) {
                    console.error("[LTXDimCalc] Failed to fetch resolutions:", err);
                }
            };

            // ------------------------------------------------------------------
            // Swap ratio widget labels to match current orientation
            // ------------------------------------------------------------------
            const updateRatioLabels = () => {
                const isPortrait  = orientationWidget.value === "Portrait";
                const labelMap    = isPortrait ? PORTRAIT_LABELS : LANDSCAPE_LABELS;
                const currentKey  = toRatioKey(ratioWidget.value);
                const newLabels   = Object.values(labelMap);

                ratioWidget.options.values = newLabels;
                ratioWidget.value = labelMap[currentKey] ?? newLabels[0];
                app.graph.setDirtyCanvas(true, true);
            };

            // ------------------------------------------------------------------
            // Wrap callbacks
            // ------------------------------------------------------------------
            const origRatioCb = ratioWidget.callback;
            ratioWidget.callback = function () {
                const r = origRatioCb?.apply(this, arguments);
                fetchResolutions();
                return r;
            };

            const origOrientationCb = orientationWidget.callback;
            orientationWidget.callback = function () {
                const r = origOrientationCb?.apply(this, arguments);
                updateRatioLabels();
                fetchResolutions();
                return r;
            };

            // Initial population — defer so node widgets are fully set up
            setTimeout(() => {
                // Restore correct label set if workflow was saved in portrait mode
                const currentKey = toRatioKey(ratioWidget.value);
                const isPortrait = orientationWidget.value === "Portrait";
                if (isPortrait && PORTRAIT_LABELS[currentKey]) {
                    ratioWidget.options.values = Object.values(PORTRAIT_LABELS);
                    ratioWidget.value = PORTRAIT_LABELS[currentKey];
                }
                fetchResolutions();
            }, 100);
        };
    },
});
