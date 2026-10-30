// In-browser inference with ONNX Runtime Web. Preprocessing mirrors src/common.py eval_tf():
// resize to 256, centre-crop 224, scale to 0-1, normalise with ImageNet mean/std.
const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225], SIZE = 224;
const $ = (id) => document.getElementById(id);
let CLASSES = ["Cat", "Dog"], session = null;

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
ort.env.wasm.numThreads = 1; // avoids needing cross-origin isolation headers

async function init() {
  try {
    const r = await fetch("model/class_names.json");
    if (r.ok) CLASSES = await r.json();
  } catch (_) { /* default labels */ }
  try {
    session = await ort.InferenceSession.create("model/catdog.onnx", { executionProviders: ["wasm"] });
    $("status").textContent = "Model ready. Add an image to classify.";
  } catch (e) {
    $("status").textContent = "The model could not be loaded. Check that web/model/catdog.onnx exists.";
    console.error(e);
  }
}

function toTensor(img) {
  const c = document.createElement("canvas");
  c.width = c.height = SIZE;
  const ctx = c.getContext("2d");
  const s = Math.min(img.naturalWidth, img.naturalHeight), crop = (s * SIZE) / 256;
  ctx.drawImage(img, (img.naturalWidth - crop) / 2, (img.naturalHeight - crop) / 2, crop, crop, 0, 0, SIZE, SIZE);
  const { data } = ctx.getImageData(0, 0, SIZE, SIZE);
  const out = new Float32Array(3 * SIZE * SIZE), n = SIZE * SIZE;
  for (let i = 0; i < n; i++)
    for (let k = 0; k < 3; k++) out[k * n + i] = (data[i * 4 + k] / 255 - MEAN[k]) / STD[k];
  return new ort.Tensor("float32", out, [1, 3, SIZE, SIZE]);
}

async function classify(img) {
  if (!session) { $("status").textContent = "The model is not loaded yet."; return; }
  const res = await session.run({ input: toTensor(img) });
  const logits = Array.from(res.logits.data), m = Math.max(...logits);
  const e = logits.map((v) => Math.exp(v - m)), sum = e.reduce((a, b) => a + b, 0);
  const p = e.map((v) => v / sum), best = p.indexOf(Math.max(...p));
  $("status").hidden = true; $("bars").hidden = false;
  $("label").textContent = "It's a " + CLASSES[best].toLowerCase();
  $("conf").textContent = (p[best] * 100).toFixed(1) + "% sure";
  ["cat", "dog"].forEach((k, i) => {
    $("bar-" + k).style.width = (p[i] * 100).toFixed(1) + "%";
    $("pc-" + k).textContent = (p[i] * 100).toFixed(1) + "%";
  });
}

function handleFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    $("status").hidden = false; $("bars").hidden = true;
    $("status").textContent = "That file is not an image. Choose a JPG, PNG or WebP photo.";
    return;
  }
  const url = URL.createObjectURL(file), img = $("preview");
  img.onload = () => { classify(img); };
  img.src = url; img.hidden = false; $("hint").hidden = true;
}

const drop = $("drop");
$("file").addEventListener("change", (e) => handleFile(e.target.files[0]));
drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("file").click(); } });
["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => handleFile(e.dataTransfer.files[0]));
window.addEventListener("paste", (e) => { const f = [...e.clipboardData.files][0]; if (f) handleFile(f); });
init();
