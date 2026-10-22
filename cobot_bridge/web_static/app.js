const state = {
  overview: null,
  selectedJob: null,
  refreshPromise: null,
  eventVersion: 0,
  eventCursor: null,
  eventController: null,
  eventsStopped: false,
  robotStatus: null,
  robotWatchdog: null,
  jointState: null,
  jointWatchdog: null,
  runtimePatch: { jobs: null, policy_server: null, rlt_policy_server: null, raws: new Map() },
  activeIlRecord: null,
  activeRlRecord: null,
  activeReplay: null,
  activeRollout: null,
  activeRltShadow: null,
  rltPolicyServer: null,
  policyServer: null,
  actionHorizonServerKey: null,
  preview: {},
};

// AGX uses its loopback service; all other Dashboard browsers default to one
// PC-side latest-JPEG relay on the same origin. The AGX marker is persisted
// per browser profile, so the PC browser never accidentally requests 127.0.0.1
// on the PC while the touchscreen keeps using its local low-latency path.
const AGX_LOCAL_MJPEG_URL = "http://127.0.0.1:8091/mjpeg";
const PC_RELAY_MJPEG_URL = "/api/live/mjpeg";
const LIVE_CAMERA_NAMES = ["front", "left", "right"];
const PREVIEW_ROUTE_STORAGE_KEY = "cobot_preview_route";
const POLICY_RUN_STORAGE_KEY = "cobot_policy_run";
const POLICY_CHECKPOINT_STORAGE_KEY = "cobot_policy_checkpoint";
const POLICY_DENOISING_STEPS_STORAGE_KEY = "cobot_policy_denoising_steps";
const REPLAY_DATASET_STORAGE_KEY = "cobot_replay_lerobot_dataset";
const REPLAY_EPISODE_STORAGE_KEY = "cobot_replay_lerobot_episode";
const previewQuery = new URLSearchParams(window.location.search);
const agxPreviewRequested = previewQuery.get("preview") === "agx-local" || previewQuery.has("agx_mjpeg");
if (agxPreviewRequested) localStorage.setItem(PREVIEW_ROUTE_STORAGE_KEY, "agx-local");
const PREVIEW_ROUTE = agxPreviewRequested || localStorage.getItem(PREVIEW_ROUTE_STORAGE_KEY) === "agx-local"
  ? "agx-local"
  : "pc-relay";
const PREVIEW_ROUTE_LABEL = PREVIEW_ROUTE === "agx-local"
  ? "AGX loopback MJPEG"
  : "PC relay MJPEG";

const $ = (id) => document.getElementById(id);

function headers() {
  const token = localStorage.getItem("cobot_web_token");
  return { "Content-Type": "application/json", ...(token ? { "X-Cobot-Token": token } : {}) };
}

async function api(path, options = {}) {
  const response = await fetch(path, { ...options, headers: { ...headers(), ...(options.headers || {}) } });
  const text = await response.text();
  let body = text;
  try { body = text ? JSON.parse(text) : null; } catch (_) {}
  if (!response.ok) throw new Error(body?.detail || body || `HTTP ${response.status}`);
  return body;
}

function toast(message, error = false) {
  const node = $("toast");
  node.textContent = message;
  node.className = `toast${error ? " error" : ""}`;
  clearTimeout(node.timer);
  node.timer = setTimeout(() => node.classList.add("hidden"), 4200);
}

function optionLabel(item, kind) {
  if (kind === "raw") return `${item.id} · ${item.collection_type || "UNKNOWN"} · ${item.frames || 0}f · ${item.outcome || "unknown"}`;
  if (kind === "dataset") return `${item.id} · ${item.episodes || 0}ep · ${item.frames || 0}f`;
  return `${item.id} · ${item.type || "unknown"}`;
}

function fillSelect(id, items, kind) {
  const select = $(id);
  const current = select.value;
  select.innerHTML = "";
  for (const item of items) {
    const option = document.createElement("option");
    option.value = item.id;
    option.textContent = optionLabel(item, kind);
    select.appendChild(option);
  }
  if ([...select.options].some((option) => option.value === current)) select.value = current;
  if (!items.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = `No ${kind} available`;
    select.appendChild(option);
  }
}

function fillLerobotReplayDatasets(datasets) {
  const select = $("replayDataset");
  const previous = select.value || localStorage.getItem(REPLAY_DATASET_STORAGE_KEY) || "";
  const preferred = "cobotmagic/5482b62ee3ff498a92f139f0788c0544_head_eef_joint";
  select.innerHTML = "";
  for (const item of datasets) {
    const option = document.createElement("option");
    option.value = item.id;
    option.textContent = `${item.id} · ${item.episodes} ep · ${item.fps} FPS`;
    select.appendChild(option);
  }
  if (!datasets.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No LeRobot replay dataset found";
    select.appendChild(option);
  } else if (datasets.some((item) => item.id === previous)) {
    select.value = previous;
  } else if (datasets.some((item) => item.id === preferred)) {
    select.value = preferred;
  } else {
    select.value = datasets[0].id;
  }

  const selected = datasets.find((item) => item.id === select.value);
  const episode = $("replayEpisode");
  const maximum = Math.max(0, Number(selected?.episodes || 1) - 1);
  episode.max = String(maximum);
  const savedEpisode = localStorage.getItem(REPLAY_EPISODE_STORAGE_KEY);
  if (episode.value === "" && savedEpisode !== null) episode.value = savedEpisode;
  const index = Number(episode.value);
  if (document.activeElement !== episode && (!Number.isInteger(index) || index < 0 || index > maximum)) {
    episode.value = "0";
  }
  localStorage.setItem(REPLAY_DATASET_STORAGE_KEY, select.value);
  localStorage.setItem(REPLAY_EPISODE_STORAGE_KEY, episode.value || "0");
  $("lerobotReplayHint").textContent = selected
    ? `右臂 action_joint（6 个关节 + 夹爪），单臂 Puppet 模式。${selected.episodes} 个 episode；先慢速对齐到所选 episode 第 0 帧，再开始回放。`
    : `没有找到可回放的 LeRobot 数据集。检查 replay 根目录：${state.overview?.paths?.lerobot_replay_root || "—"}`;
}

function renderReplaySourceControls() {
  const lerobot = $("replaySource").value === "lerobot";
  $("lerobotReplayControls").hidden = !lerobot;
  $("rawReplayControls").hidden = lerobot;
  $("replayLoopControls").hidden = lerobot;
}

function rawCollectionId(raw) {
  if (raw.collection) return raw.collection;
  const parts = raw.id.split("/");
  return parts.length > 1 ? parts.slice(0, -1).join("/") : ".";
}

function renderRawEpisodeDetail(selected) {
  $("episodeDetailTitle").textContent = selected?.episode || "No episode selected";
  $("episodeDetailTask").textContent = selected?.task || "—";
  $("episodeDetailSetup").textContent = selected?.task_setup_label || selected?.task_setup || "Unlabeled";
  $("episodeDetailObjective").textContent = selected?.task_objective || selected?.task || "—";
  $("episodeDetailOutcome").textContent = selected?.outcome || "—";
  $("episodeDetailType").textContent = selected?.collection_type || "—";
  $("episodeDetailFrames").textContent = selected
    ? `${selected.frames || 0} frames / ${selected.fps || 0} FPS`
    : "—";
  $("episodeDetailSize").textContent = selected
    ? `${(Number(selected.size || 0) / 1048576).toFixed(1)} MiB`
    : "—";
  $("episodeDetailPath").textContent = selected?.id || "—";
}

function renderRawEpisodeList(raws, collection, preferredRawId = "") {
  const list = $("rawEpisodeList");
  const items = raws
    .filter((raw) => rawCollectionId(raw) === collection)
    .sort((left, right) => Number(right.modified_at_ns || 0) - Number(left.modified_at_ns || 0)
      || right.id.localeCompare(left.id, undefined, { numeric: true }));
  const selected = items.find((item) => item.id === preferredRawId) || items[0] || null;
  $("convertRaw").value = selected?.id || "";
  $("episodeListCount").textContent = `${items.length} episode${items.length === 1 ? "" : "s"}`;
  list.innerHTML = "";
  for (const [index, item] of items.entries()) {
    const row = document.createElement("button");
    row.type = "button";
    row.className = `episode-row${item.id === selected?.id ? " selected" : ""}`;
    const title = document.createElement("strong");
    title.textContent = `${item.episode || item.id.split("/").at(-1)}${index === 0 ? " · Latest" : ""}`;
    const outcome = document.createElement("span");
    const outcomeValue = item.outcome || "unknown";
    outcome.className = `episode-outcome${["success", "failure"].includes(outcomeValue) ? ` ${outcomeValue}` : ""}`;
    outcome.textContent = outcomeValue;
    const metadata = document.createElement("small");
    metadata.textContent = `${item.task_setup_label || item.task_setup || "Unlabeled"} · ${item.frames || 0}f · ${item.fps || 0} FPS`;
    row.append(title, outcome, metadata);
    row.onclick = () => renderRawEpisodeList(raws, collection, item.id);
    list.appendChild(row);
  }
  if (!items.length) list.innerHTML = '<p class="empty">No completed episodes.</p>';
  renderRawEpisodeDetail(selected);
}

function fillRawSelectors(raws) {
  const collectionSelect = $("convertRawCollection");
  const currentCollection = collectionSelect.value;
  const currentRaw = $("convertRaw").value;
  const groups = new Map();
  for (const raw of raws) {
    const collection = rawCollectionId(raw);
    if (!groups.has(collection)) groups.set(collection, []);
    groups.get(collection).push(raw);
  }
  const collections = [...groups.keys()].sort((left, right) => {
    const leftLatest = Math.max(...groups.get(left).map((raw) => Number(raw.modified_at_ns || 0)));
    const rightLatest = Math.max(...groups.get(right).map((raw) => Number(raw.modified_at_ns || 0)));
    return rightLatest - leftLatest || left.localeCompare(right, undefined, { numeric: true });
  });
  const selectedCollection = groups.has(currentCollection) ? currentCollection : collections[0] || "";
  collectionSelect.innerHTML = "";
  for (const collection of collections) {
    const items = groups.get(collection);
    const outcomes = items.reduce((counts, item) => {
      const key = item.outcome || "unknown";
      counts[key] = (counts[key] || 0) + 1;
      return counts;
    }, {});
    const summary = Object.entries(outcomes).map(([key, count]) => `${count} ${key}`).join(" / ");
    const tasks = [...new Set(items.map((item) => item.task).filter(Boolean))];
    const taskSummary = tasks.length === 1 ? tasks[0] : `${tasks.length} tasks`;
    const setups = [...new Set(items.map((item) => item.task_setup_label || item.task_setup).filter(Boolean))];
    const setupSummary = setups.length === 1 ? setups[0] : `${setups.length} task setups`;
    const option = document.createElement("option");
    option.value = collection;
    option.textContent = `${collection} · ${setupSummary} · ${taskSummary} · ${items.length} episode${items.length === 1 ? "" : "s"} · ${summary}`;
    collectionSelect.appendChild(option);
  }
  if (selectedCollection) collectionSelect.value = selectedCollection;
  if (!collections.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No raw collection available";
    collectionSelect.appendChild(option);
  }
  renderRawEpisodeList(raws, selectedCollection, currentRaw);
}

function policyRunId(policy) {
  if (policy.run_id) return policy.run_id;
  const marker = "/checkpoints/";
  return policy.id.includes(marker) ? policy.id.split(marker)[0] : policy.id;
}

function updatePolicySelectOptions(select, options) {
  const existing = [...select.options].map((option) => [option.value, option.textContent]);
  if (JSON.stringify(existing) === JSON.stringify(options)) return;
  select.innerHTML = "";
  for (const [value, label] of options) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    select.appendChild(option);
  }
}

function fillCheckpointSelect(policies, runId, preferredPolicyId = "") {
  const select = $("policySelect");
  const items = policies
    .filter((policy) => policyRunId(policy) === runId)
    .sort((left, right) => {
      const leftStep = left.step !== null && left.step !== undefined && Number.isFinite(Number(left.step)) ? Number(left.step) : -1;
      const rightStep = right.step !== null && right.step !== undefined && Number.isFinite(Number(right.step)) ? Number(right.step) : -1;
      return rightStep - leftStep || right.id.localeCompare(left.id, undefined, { numeric: true });
    });
  const options = items.map((item, index) => {
    const checkpoint = item.checkpoint || item.id.split("/").at(-2) || item.id;
    return [item.id, `${checkpoint}${index === 0 ? " · Latest" : ""} · ${item.type || "unknown"}`];
  });
  updatePolicySelectOptions(select, options.length ? options : [["", "No checkpoint available"]]);
  if ([...select.options].some((option) => option.value === preferredPolicyId)) {
    select.value = preferredPolicyId;
  }
}

function fillPolicySelectors(policies, loadedPolicyId = "") {
  const runSelect = $("policyRunSelect");
  const currentRun = runSelect.value;
  const currentPolicy = $("policySelect").value;
  const savedRun = localStorage.getItem(POLICY_RUN_STORAGE_KEY) || "";
  const savedPolicy = localStorage.getItem(POLICY_CHECKPOINT_STORAGE_KEY) || "";
  const groups = new Map();
  for (const policy of policies) {
    const runId = policyRunId(policy);
    if (!groups.has(runId)) groups.set(runId, []);
    groups.get(runId).push(policy);
  }
  const runs = [...groups.keys()].sort((left, right) => left.localeCompare(right, undefined, { numeric: true }));
  const loaded = policies.find((policy) => policy.id === loadedPolicyId);
  const selectedRun = groups.has(currentRun)
      ? currentRun
      : groups.has(savedRun)
        ? savedRun
        : loaded ? policyRunId(loaded) : runs[0] || "";
  const options = [];
  for (const runId of runs) {
    const items = groups.get(runId);
    const steps = items
      .filter((item) => item.step !== null && item.step !== undefined)
      .map((item) => Number(item.step))
      .filter(Number.isFinite);
    const latest = steps.length ? Math.max(...steps) : null;
    options.push([runId, `${runId} · ${items.length} ckpt${items.length === 1 ? "" : "s"}${latest === null ? "" : ` · latest ${latest}`}`]);
  }
  updatePolicySelectOptions(runSelect, options.length ? options : [["", "No training run available"]]);
  if (selectedRun) runSelect.value = selectedRun;
  fillCheckpointSelect(policies, selectedRun, currentPolicy || savedPolicy || loadedPolicyId);
}

function policySelectionMismatch(server) {
  const selected = $("policySelect").value;
  return Boolean(server?.ready && server.policy_id && selected && selected !== server.policy_id);
}

function fillRltCheckpointSelector(checkpoints, serverCheckpointId = "") {
  const select = $("rltCheckpointSelect");
  const current = select.value;
  select.innerHTML = "";
  for (const [index, checkpoint] of checkpoints.entries()) {
    const option = document.createElement("option");
    option.value = checkpoint.id;
    const step = checkpoint.final_step ? `step ${checkpoint.final_step}` : "unknown step";
    const loss = Number.isFinite(Number(checkpoint.final_loss))
      ? `loss ${Number(checkpoint.final_loss).toFixed(4)}`
      : "loss unavailable";
    option.textContent = `${checkpoint.run_id || checkpoint.id} · ${step} · ${loss}${index === 0 ? " · Latest" : ""}`;
    select.appendChild(option);
  }
  const preferred = serverCheckpointId || current;
  if ([...select.options].some((option) => option.value === preferred)) select.value = preferred;
  if (!checkpoints.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No Stage 1 checkpoint available";
    select.appendChild(option);
  }
}

function selectedRltCheckpoint(overview) {
  return (overview.rlt?.stage1_checkpoints || []).find(
    (checkpoint) => checkpoint.id === $("rltCheckpointSelect").value
  ) || null;
}

function fillRltActorCheckpointSelector(checkpoint, serverActorCheckpointId = "") {
  const select = $("rltActorCheckpointSelect");
  const current = select.value;
  const actors = checkpoint?.actor_checkpoints || [];
  select.innerHTML = "";
  for (const actor of actors) {
    const option = document.createElement("option");
    option.value = actor.id;
    const loss = Number.isFinite(Number(actor.validation_gt_bc_loss))
      ? ` · val ${Number(actor.validation_gt_bc_loss).toFixed(5)}`
      : "";
    const step = actor.bc_steps ? ` · BC step ${actor.bc_steps}` : "";
    option.textContent = actor.read_only
      ? `${actor.label}`
      : `${actor.label} · updates ${actor.total_updates}${step}${loss}`;
    select.appendChild(option);
  }
  const currentExists = [...select.options].some((option) => option.value === current);
  const serverExists = [...select.options].some(
    (option) => option.value === serverActorCheckpointId
  );
  if (serverExists) {
    // A resident RLT server owns the model selection. Restore that exact Actor
    // after a browser refresh instead of falling back to the first dropdown item.
    select.value = serverActorCheckpointId;
  } else if (currentExists) {
    select.value = current;
  } else {
    const latest = actors.find((actor) => actor.follows_latest);
    if (latest) select.value = latest.id;
  }
  if (!actors.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No compatible Actor checkpoint";
    select.appendChild(option);
  }
}

function selectedRltActor(overview) {
  const checkpoint = selectedRltCheckpoint(overview);
  return (checkpoint?.actor_checkpoints || []).find(
    (actor) => actor.id === $("rltActorCheckpointSelect").value
  ) || null;
}

function formatDuration(seconds) {
  if (seconds === null || seconds === undefined || seconds === "") return "—";
  const value = Number(seconds);
  if (!Number.isFinite(value) || value < 0) return "—";
  const rounded = Math.max(0, Math.round(value));
  const hours = Math.floor(rounded / 3600);
  const minutes = Math.floor((rounded % 3600) / 60);
  const remaining = rounded % 60;
  if (hours) return `${hours}h ${String(minutes).padStart(2, "0")}m`;
  if (minutes) return `${minutes}m ${String(remaining).padStart(2, "0")}s`;
  return `${remaining}s`;
}

function renderRltTrainerProgress(pipeline) {
  const panel = $("rltTrainerProgressPanel");
  const progress = pipeline?.trainer_progress;
  const validProgress = progress && progress.state !== "not_started";
  if (!validProgress) {
    panel.className = "rlt-trainer-progress idle";
    $("rltTrainerProgressTitle").textContent = "Trainer idle · waiting for a labeled actor episode";
    $("rltTrainerEta").textContent = "ETA —";
    $("rltTrainerProgressBar").style.width = "0%";
    panel.querySelector(".rlt-progress-track").setAttribute("aria-valuenow", "0");
    $("rltTrainerProgressText").textContent = "0/0 · 0.0% · — upd/s";
    $("rltTrainerMetrics").textContent = "Each saved actor episode automatically requests an update";
    return;
  }

  const trainerState = progress.state || "running";
  const current = Number(progress.current_update || 0);
  const total = Number(progress.total_updates || 0);
  const percent = Math.min(100, Math.max(0, Number(
    progress.percent ?? (total ? 100 * current / total : 0)
  )));
  const rate = progress.updates_per_second === null || progress.updates_per_second === undefined
    ? Number.NaN
    : Number(progress.updates_per_second);
  const metrics = progress.last_metrics || {};
  const metricText = (name) => Number.isFinite(Number(metrics[name]))
    ? Number(metrics[name]).toFixed(5)
    : "—";
  const phase = "Async actor–critic";
  panel.className = `rlt-trainer-progress ${trainerState}`;
  $("rltTrainerProgressTitle").textContent = trainerState === "completed"
    ? `${phase} complete`
    : trainerState === "failed" ? `${phase} failed` : `${phase} training`;
  $("rltTrainerEta").textContent = trainerState === "completed"
    ? `Done · ${formatDuration(progress.elapsed_seconds)}`
    : trainerState === "failed"
      ? "ETA unavailable"
      : `ETA ${formatDuration(progress.eta_seconds)}`;
  $("rltTrainerProgressBar").style.width = `${percent}%`;
  panel.querySelector(".rlt-progress-track").setAttribute("aria-valuenow", percent.toFixed(1));
  $("rltTrainerProgressText").textContent = `${current}/${total} · ${percent.toFixed(1)}% · ${Number.isFinite(rate) ? rate.toFixed(2) : "—"} upd/s`;
  $("rltTrainerMetrics").textContent = trainerState === "failed"
    ? progress.error || "trainer exited with an error"
    : `critic ${metricText("critic_loss")} · actor ${metricText("actor_loss")} · elapsed ${formatDuration(progress.elapsed_seconds)}`;
}

function renderRltControls(overview) {
  const rltServer = overview.rlt?.policy_server || { ready: false, state: "offline" };
  // A resident RLT server owns one immutable Stage 1. Never leave a stale
  // browser selection visible while its selector is locked: it falsely shows
  // "server and selected Stage 1 do not match" even when the loaded model is
  // present in the inventory.
  const rltCheckpointSelect = $("rltCheckpointSelect");
  if (rltServer.ready && rltServer.checkpoint_id
      && [...rltCheckpointSelect.options].some((option) => option.value === rltServer.checkpoint_id)) {
    rltCheckpointSelect.value = rltServer.checkpoint_id;
  }
  const selected = selectedRltCheckpoint(overview);
  const serverActorCheckpointId = rltServer.checkpoint_id === selected?.id
    ? rltServer.actor_checkpoint_id || ""
    : "";
  fillRltActorCheckpointSelector(selected, serverActorCheckpointId);
  const selectedActor = selectedRltActor(overview);
  state.rltPolicyServer = rltServer;
  const rltSmoothProfile = $("rltSmoothProfile");
  const eefRlt = rltServer.action_space === "eef_target";
  for (const option of rltSmoothProfile.options) {
    option.disabled = eefRlt && !["edge_v4", "raw"].includes(option.value);
  }
  if (eefRlt && !["edge_v4", "raw"].includes(rltSmoothProfile.value)) {
    rltSmoothProfile.value = "edge_v4";
  }
  const activeShadow = overview.jobs.find((job) =>
    job.kind === "rlt_shadow" && ["running", "stopping"].includes(job.state)
  ) || null;
  const activeValidation = overview.jobs.find((job) =>
    job.kind === "rlt_validate" && ["running", "stopping"].includes(job.state)
  ) || null;
  state.activeRltShadow = activeShadow;
  const activeOnline = overview.jobs.find((job) =>
    job.kind === "rlt_online_execute" && ["running", "stopping"].includes(job.state)
  ) || null;
  const latestOnline = overview.jobs.find((job) => job.kind === "rlt_online_execute") || null;
  state.activeRltOnline = activeOnline;
  state.latestRltOnline = latestOnline;
  if (state.teachLabelJobId !== (latestOnline && latestOnline.id)) {
    state.teachLabelJobId = latestOnline ? latestOnline.id : null;
    $("teachSegmentList").innerHTML = "";
    $("saveTeachLabelsButton").disabled = true;
  }
  $("loadTeachSegmentsButton").disabled = !latestOnline;
  $("rltCheckpointStatus").textContent = selected
    ? selected.ready ? "Ready" : "Incomplete"
    : "No checkpoint";
  const vlm = selected?.embedding_shapes?.vlm;
  const expert = selected?.embedding_shapes?.expert;
  $("rltEmbeddingShape").textContent = selected
    ? `VLM ${vlm?.join("×") || "?"} · Expert ${expert?.join("×") || "?"}`
    : "—";
  $("rltTokenShape").textContent = selected
    ? `${selected.rlt_hidden_dim || "?"}D · chunk ${selected.chunk_size || "?"}`
    : "—";
  $("rltTrainingSummary").textContent = selected
    ? `${selected.dataset_episodes || 0} ep · ${selected.dataset_frames || 0} frames · step ${selected.final_step || "?"} · loss ${Number(selected.final_loss).toFixed(4)}`
    : "—";
  $("rltBasePolicy").textContent = selected?.smolvla_path || "—";
  const rltServerLocked = rltServer.ready || rltServer.state === "loading";
  $("rltCheckpointSelect").disabled = Boolean(
    activeShadow || activeValidation || activeOnline || rltServerLocked
  );
  $("rltActorCheckpointSelect").disabled = Boolean(
    activeOnline || rltServerLocked || !selectedActor
  );
  $("validateRltButton").disabled = !selected?.ready || selected?.inference_only || Boolean(activeValidation || activeShadow);
  $("validateRltButton").textContent = activeValidation ? "Validating…" : "Validate Stage 1";
  const serverBusy = overview.policy_server?.compute !== "robot"
    && (overview.policy_server?.ready || overview.policy_server?.state === "loading");
  const rltServerBusy = rltServer.ready || rltServer.state === "loading";
  const shadowReady = selected?.ready && !selected?.inference_only && overview.bridge.online && overview.bridge.devices?.ok
    && !serverBusy && !rltServerBusy;
  $("startRltShadowButton").disabled = !shadowReady || Boolean(activeShadow);
  $("stopRltShadowButton").disabled = !activeShadow;
  $("rltShadowHint").textContent = activeShadow
    ? `RLT shadow is ${activeShadow.state}; action_sent=false. Diagnostics: ${activeShadow.metadata?.diagnostic_output || "job log"}`
    : serverBusy
      ? "Stop the normal model server before starting RLT shadow on PC GPU 0."
      : rltServerBusy
        ? "Stop the persistent RLT rollout server before starting read-only shadow."
      : !overview.bridge.online || !overview.bridge.devices?.ok
        ? "Bridge and all required arm/camera observations must be online."
        : "PC GPU 0 · read-only bridge observations · action_sent=false.";

  const rltServerMatches = Boolean(
    rltServer.ready && selected?.id && selectedActor?.id
    && rltServer.checkpoint_id === selected.id
    && rltServer.actor_checkpoint_id === selectedActor.id
  );
  const stage2 = selected?.stage2 || {
    actor_ready: false,
  };
  const actorReady = Boolean(selectedActor?.ready);
  const actorReadOnly = Boolean(
    selectedActor?.read_only || (rltServer.ready && rltServer.actor_read_only)
  );
  $("rltExplorationStd").disabled = actorReadOnly;
  if (actorReadOnly) $("rltExplorationStd").value = "0";
  const qSelectEl = $("rltQSelect");
  if (qSelectEl) {
    qSelectEl.disabled = actorReadOnly;
    if (actorReadOnly) qSelectEl.value = "0";
  }
  const rolloutControlsLocked = Boolean(activeOnline || rltServer.active_episode);
  $("rltActorEefGate").disabled = rolloutControlsLocked;
  $("rltSmoothProfile").disabled = rolloutControlsLocked;
  $("rltServerBadge").textContent = rltServer.ready
    ? "Ready"
    : rltServer.state === "loading" ? "Loading" : "Offline";
  $("rltServerBadge").className = `badge ${rltServer.ready ? "online" : "offline"}`;
  $("rltServerEndpoint").textContent = rltServer.endpoint
    || overview.paths?.rlt_policy_server_endpoint || "tcp://127.0.0.1:5561";
  $("rltServerStage1").textContent = rltServer.checkpoint_id || "—";
  $("rltServerActor").textContent = rltServer.ready
    ? rltServer.actor_ready
      ? rltServer.actor_read_only
        ? "STEAM offline actor · deterministic · no async training"
        : `${rltServer.actor_total_updates || 0} updates · generation ${rltServer.actor_generation || 0}${rltServer.actor_update_pending ? " · update pending" : ""}`
      : actorReady
        ? `${selectedActor?.label || "trained checkpoint"} ready${actorReadOnly ? " · deterministic · no async training" : ` · generation ${rltServer.actor_generation || 0}`}`
        : rltServer.actor_update_pending
          ? `checkpoint pending next episode · generation ${rltServer.actor_generation || 0}`
          : `actor checkpoint unavailable · generation ${rltServer.actor_generation || 0}`
    : "—";
  $("rltServerEpisode").textContent = rltServer.active_episode
    ? `active · actor ${rltServer.actor_enabled ? "on" : "off"}`
    : activeOnline
      ? "episode process starting · first action chunk takes 10-15 s after a cold server load"
      : "idle";
  const canStartRltServer = Boolean(
    selected?.ready && selectedActor?.ready && overview.bridge.online && overview.bridge.devices?.ok
    && !serverBusy && !activeShadow && !activeOnline && !rltServerLocked
  );
  $("startRltServerButton").disabled = !canStartRltServer;
  $("stopRltServerButton").disabled = !rltServerLocked || Boolean(activeOnline || rltServer.active_episode);
  const stage2Training = stage2.pipeline
    && ["starting", "waiting_for_trainer_lock", "running"].includes(stage2.pipeline.state);
  const rollbackAvailable = Boolean(stage2.rollback_30k_available);
  const rollbackActive = Boolean(stage2.rollback_30k_active)
    && (!rltServerMatches || Number(rltServer.actor_total_updates || 0) === 30000);
  const rollbackBlocked = Boolean(
    activeOnline || rltServer.active_episode || stage2Training
    || overview.bridge.teach_active || overview.bridge.output_enabled
  );
  $("rollbackRlt30kButton").disabled = !rollbackAvailable || rollbackActive || rollbackBlocked;
  $("rollbackRlt30kButton").textContent = rollbackActive
    ? "30k Actor Active"
    : stage2Training ? "Training in progress…" : "Rollback Actor to 30k";
  $("rollbackRlt30kHint").textContent = !rollbackAvailable
    ? "checkpoint_00030000.pt is unavailable for this Stage 1 run."
    : rollbackActive
      ? "The published checkpoint and resident rollout server are both on 30k."
      : rollbackBlocked
        ? "Finish the episode/training and turn Teach + robot output off before rollback."
        : "Restores checkpoint_00030000.pt atomically; no robot command is sent.";
  $("rltServerHint").textContent = rltServer.ready
    ? rltServerMatches
      ? actorReadOnly
        ? "STEAM offline actor loaded. Exploration is forced to 0; this rollout only records results and never starts async training."
        : rltServer.actor_update_pending
        ? "A newer actor is published; it will load atomically at the next episode boundary."
        : "Server stays loaded. Start Mode 1 or Mode 2; stopping an active episode is blocked."
      : "Loaded Stage 1 differs from the selector; stop the server before changing models."
    : rltServer.state === "loading"
      ? "Loading frozen SmolVLA + Stage 1 encoder on GPU 0…"
      : serverBusy
        ? "Stop the normal PC model server to free GPU 0."
        : !overview.bridge.online || !overview.bridge.devices?.ok
          ? "Bridge hardware must be 7/7 before starting the RLT server."
          : selectedActor?.ready
            ? `Selected ${selectedActor.label}${actorReadOnly || selected?.inference_only ? " · inference-only; no asynchronous update" : ` · ${selectedActor.total_updates} updates`}.`
            : "Select a compatible Stage 1 and Actor checkpoint.";

  const onlineReady = Boolean(overview.rlt?.online_stage2_ready && rltServerMatches);
  $("rltOnlineBadge").textContent = !onlineReady
    ? "Locked"
    : actorReady ? "Actor Ready" : "Stage 2 Required";
  $("rltOnlineBadge").className = `badge ${onlineReady ? "online" : "offline"}`;
  $("rltOnlineReason").textContent = !overview.rlt?.online_stage2_ready
    ? overview.rlt?.online_stage2_reason || "Stage 2 unavailable"
    : !rltServer.ready
      ? "Start the persistent RLT rollout server first."
        : !rltServerMatches
        ? "The server and selected Stage 1 checkpoint do not match."
        : actorReady
          ? actorReadOnly
            ? "STEAM offline actor ready · deterministic exploration=0 · no async training."
            : `${selectedActor.label} ready · ${selectedActor.total_updates} updates.`
          : "Train Stage 2 once from the available labeled replay.";
  const baseStartReady = onlineReady
    && selected?.ready
    && overview.bridge.online
    && overview.bridge.devices?.ok
    && overview.bridge.allow_control
    && !overview.bridge.teach_active
    && !overview.bridge.emergency_latched
    && !activeOnline
    && !activeShadow
    && !rltServer.active_episode;
  const actorStartReady = baseStartReady && actorReady;
  $("rltActorGate").textContent = actorReady
    ? actorReadOnly
      ? "STEAM offline actor · Mode 1/2 unlocked · exploration=0"
      : `${selectedActor?.total_updates || 0} updates · Mode 1/2 unlocked`
    : "Locked · train Stage 2 once from available replay";
  $("startRltMode1Button").disabled = !actorStartReady;
  $("startRltMode2Button").disabled = !actorStartReady;
  $("startRltReferenceButton").disabled = !baseStartReady;
  const resultSubmitted = Boolean(activeOnline?.metadata?.result_submitted);
  const livePipeline = latestOnline?.metadata?.checkpoint_id === selected?.id
    ? latestOnline?.metadata?.pipeline
    : null;
  const pipeline = livePipeline || stage2.pipeline || null;
  const failedBeforeUpdates = Number(pipeline?.trainer_progress?.global_updates_before || 0);
  const pipelineRecovered = pipeline?.state === "failed"
    && Number(stage2.total_updates || 0) > failedBeforeUpdates;
  const displayedPipeline = pipelineRecovered
    ? {
        ...pipeline,
        state: "completed",
        trainer_progress: {
          ...(pipeline.trainer_progress || {}),
          state: "completed",
          current_update: Number(pipeline.trainer_progress?.total_updates || 0),
          global_update: Number(stage2.total_updates || 0),
          percent: 100,
          eta_seconds: 0,
          error: null,
        },
      }
    : pipeline;
  renderRltTrainerProgress(displayedPipeline);
  $("rltPipelineBadge").textContent = activeOnline
      ? resultSubmitted
        ? "Processing"
      : `Actor Mode ${activeOnline.metadata?.workflow_mode} live`
    : pipelineRecovered
      ? "Recovered · Actor published"
    : pipeline && !["completed", "not_started"].includes(pipeline.state)
      ? `Pipeline ${pipeline.state}`
    : onlineReady ? "Ready" : "Not connected";
  $("rltPipelineBadge").className = `badge ${onlineReady ? "online" : "offline"}`;
  $("rltPipelineHint").textContent = activeOnline
    ? resultSubmitted
      ? selected?.inference_only
        ? `Robot output is off. The inference-only episode is being finalized without actor training. Job ${activeOnline.id}.`
        : `Robot output is off. Updating actor/critic and archiving data. Job ${activeOnline.id}.`
      : `Recording actor rollout. Actor enabled=${Boolean(activeOnline.metadata?.actor_enabled)}; choose a terminal result when done.`
    : pipelineRecovered
      ? `The failed update was recovered manually; Current RL Actor update ${stage2.total_updates || 0} is published and will load at the next episode boundary.`
    : pipeline?.state === "completed"
      ? `Last pipeline completed: trainer ${pipeline.trainer_device}, archive ${pipeline.archive_root}. The next episode loads the latest published actor.`
    : pipeline?.state === "failed"
      ? `Last pipeline failed. Inspect ${latestOnline?.metadata?.pipeline_log || pipeline?.status_file || "the pipeline log"}; robot output remains off.`
    : pipeline && pipeline.state !== "not_started"
      ? `Post-episode pipeline is ${pipeline.state}: trainer ${pipeline.trainer_device}, robot output is off.`
    : onlineReady
      ? actorReady
        ? "Actor episode label triggers replay → GPU 1 update + archive. GPU 0 server remains loaded."
        : "Train Stage 2 once from the available labeled replay before real actor rollout."
      : "Current status: Stage 2 adapter is not connected, so no automatic recording, conversion, replay insertion, or training is running.";
  const canLabel = Boolean(
    activeOnline
    && ["actor", "reference"].includes(activeOnline.metadata?.rollout_phase)
    && !resultSubmitted
    && activeOnline.state === "running"
  );
  $("saveRltSuccessButton").disabled = !canLabel;
  $("saveRltFailureButton").disabled = !canLabel;
  $("abortRltButton").disabled = !canLabel;
  for (const id of ["phaseCalibrateButton", "phaseRescueButton", "phaseRetreatButton"]) $(id).disabled = !canLabel;
  if (activeOnline && state.phaseMarkJobId !== activeOnline.id) {
    state.phaseMarkJobId = activeOnline.id;
    $("phaseCalibrateButton").checked = true;
    if (canLabel) void markInterventionPhase("calibrate", true);
  }
  $("rltOutcomeHint").textContent = activeOnline
    ? resultSubmitted
      ? `Result ${activeOnline.metadata?.result || "submitted"}; post-episode jobs are running with robot output disabled.`
      : activeOnline.metadata?.workflow_mode === 1
        ? "Mode 1 has no Teach/intervention data. Choose Success or Failure for the autonomous episode."
        : "Mode 2 Teach intervals are intervention transitions. Choose Success or Failure for the complete episode."
    : actorReady
      ? "Start Actor Mode 1 or Mode 2, then choose the terminal episode reward."
      : "Actor Mode 1/2 unlocks after the first validated Stage 2 checkpoint is ready.";
}

function renderBridge(bridge) {
  const badge = $("bridgeBadge");
  badge.className = `badge ${bridge.online ? "online" : "offline"}`;
  badge.textContent = bridge.online ? "Bridge online" : "Bridge offline";
  $("modeValue").textContent = bridge.mode ?? bridge.mode_id ?? "—";
  $("teachValue").textContent = bridge.teach_active === undefined ? "—" : bridge.teach_active ? "ACTIVE" : "off";
  $("outputValue").textContent = bridge.output_enabled === undefined ? "—" : bridge.output_enabled ? "ENABLED" : "off";
  $("controlValue").textContent = bridge.emergency_latched
    ? "STOP LATCHED"
    : bridge.allow_control === undefined ? "—" : bridge.allow_control ? "LIVE" : "read only";
  renderDevices(bridge.devices || null);
}

function formatAge(value) {
  return Number.isFinite(Number(value)) ? `${Number(value).toFixed(0)} ms` : "—";
}

function liveSuffix(name) {
  return name[0].toUpperCase() + name.slice(1);
}

function previewStreamUrl(name) {
  const base = PREVIEW_ROUTE === "agx-local" ? AGX_LOCAL_MJPEG_URL : PC_RELAY_MJPEG_URL;
  const url = new URL(base, window.location.href);
  url.searchParams.set("camera", name);
  url.searchParams.set("session", String(Date.now()));
  if (PREVIEW_ROUTE === "pc-relay") {
    const token = localStorage.getItem("cobot_web_token");
    if (token) url.searchParams.set("token", token);
  }
  return url.toString();
}

function renderPreviewRoute() {
  const badge = $("previewRouteValue");
  if (badge) badge.textContent = `Preview: ${PREVIEW_ROUTE_LABEL}`;
  document.querySelectorAll(".live-tag").forEach((tag) => {
    tag.textContent = `LIVE · ${PREVIEW_ROUTE_LABEL.toUpperCase()}`;
  });
}

function startMjpegCamera(name) {
  const suffix = liveSuffix(name);
  const image = $(`camera${suffix}Live`);
  if (!image || state.preview[name]) return;
  const session = { done: false, retryTimer: null };
  state.preview[name] = session;

  const cleanup = (why) => {
    if (session.done) return;
    session.done = true;
    if (state.preview[name] === session) delete state.preview[name];
    if (session.retryTimer) clearTimeout(session.retryTimer);
    image.classList.remove("live");
    image.removeAttribute("src");
    if (why) console.warn(`[mjpeg] ${name} stopped: ${why}`);
  };

  const reconnect = (why, delayMs = 1000) => {
    if (session.done) return;
    image.classList.remove("live");
    image.removeAttribute("src");
    console.warn(`[mjpeg] ${name} reconnect: ${why}`);
    session.retryTimer = setTimeout(() => {
      if (!session.done) connect();
    }, delayMs);
  };

  const connect = () => {
    if (session.done) return;
    // A cache-busting query makes a reconnect create a fresh, latest-only
    // multipart stream rather than reusing a stalled HTTP response.
    // Multipart JPEG is one never-ending HTTP response.  Chrome does not
    // reliably emit ``load`` for each part, so show the element as soon as
    // the stream is attached; ``error`` still hides it on a real failure.
    image.classList.add("live");
    image.onload = () => {};
    image.onerror = () => reconnect("stream error", 1200);
    image.src = previewStreamUrl(name);
  };
  connect();
}

function renderDevices(devices) {
  const empty = { ok: false, ok_count: 0, total: 7, arms: {}, cameras: {} };
  const value = devices || empty;
  const summary = $("deviceSummary");
  summary.textContent = `${value.ok_count || 0} / ${value.total || 7} OK`;
  summary.className = `badge ${value.ok ? "online" : "offline"}`;
  $("hardwareValue").textContent = `${value.ok_count || 0}/${value.total || 7} ${value.ok ? "OK" : "CHECK"}`;
  $("hardwareValue").className = value.ok ? "hardware-ok" : "hardware-check";

  for (const name of ["front", "left", "right"]) {
    const camera = value.cameras?.[name] || {};
    const suffix = name[0].toUpperCase() + name.slice(1);
    const card = document.querySelector(`[data-camera-card="${name}"]`);
    card.classList.toggle("ok", Boolean(camera.ok));
    $(`camera${suffix}Status`).textContent = camera.ok ? "OK" : camera.fresh ? "NO FRAME" : "STALE";
    const viaMjpeg = Boolean(state.preview[name]) && camera.fresh && !camera.has_frame;
    $(`camera${suffix}Meta`).textContent = camera.has_frame
      ? `${formatAge(camera.age_ms)} · ${(camera.bytes / 1024).toFixed(0)} KiB · ${camera.encoding || "jpeg"}`
      : viaMjpeg
        ? `via ${PREVIEW_ROUTE_LABEL} · ${formatAge(camera.age_ms)}`
        : "No frame";
  }

  const armIds = {
    master_left: "MasterLeft",
    master_right: "MasterRight",
    puppet_left: "PuppetLeft",
    puppet_right: "PuppetRight",
  };
  for (const [key, suffix] of Object.entries(armIds)) {
    const arm = value.arms?.[key] || {};
    const card = document.querySelector(`[data-arm-card="${key}"]`);
    card.classList.toggle("ok", Boolean(arm.ok));
    $(`arm${suffix}Status`).textContent = arm.ok ? "OK" : arm.fresh ? "INVALID" : "STALE";
    if (!state.jointState) $(`arm${suffix}Meta`).textContent = "等待实时关节反馈…";
  }
  if (state.jointState) renderJointAngles();
}

function renderJointAngles() {
  const arms = state.jointState?.arms || {};
  for (const [key, suffix] of Object.entries({ master_left: "MasterLeft", master_right: "MasterRight",
    puppet_left: "PuppetLeft", puppet_right: "PuppetRight" })) {
    const arm = arms[key];
    if (!arm || arm.position?.length !== 7) continue;
    const fresh = arm.fresh && !state.jointState.stale && state.robotStatus?.online !== false;
    $(`arm${suffix}Meta`).textContent = [
      fresh ? `低频实时 · ${formatAge(arm.age_ms)}` : "反馈已过期（以下为最后收到的数值）",
      ...arm.position.slice(0, 6).map((q, i) => `J${i + 1}       ${Number(q).toFixed(3)} rad`),
      `Gripper  ${Number(arm.position[6]).toFixed(3)} m`,
    ].join("\n");
  }
}

function jointStateStale() {
  clearTimeout(state.jointWatchdog);
  if (state.jointState) { state.jointState.stale = true; renderJointAngles(); }
}

function handleJointState(data) {
  state.jointState = { ...data, stale: data.age_ms === null || data.age_ms > 1500 };
  renderJointAngles(); // Display only: no directory scans or motion-button updates.
  clearTimeout(state.jointWatchdog);
  state.jointWatchdog = setTimeout(jointStateStale, 2000);
}

function renderCapabilities(capabilities) {
  const names = {
    record: "Raw recorder", convert: "LeRobot convert", replay: "Replay baseline",
    lerobot_policy: "LeRobot policy", rlt_stage1: "RLT Stage 1",
    rlt_stage2: "RLT Stage 2", rlt_policy: "RLT shadow",
  };
  $("capabilityList").innerHTML = Object.entries(capabilities).map(([key, value]) =>
    `<span class="capability ${value.ready ? "ready" : ""}" title="${value.reason || "ready"}">${value.ready ? "✓" : "○"} ${names[key] || key}</span>`
  ).join("");
}

function activateTab(name) {
  document.body.classList.toggle("document-scroll", ["il", "rl", "rlt", "data", "system"].includes(name));
  document.querySelectorAll(".module-tab").forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.tab === name);
  });
  document.querySelectorAll(".tab-panel").forEach((panel) => {
    panel.classList.toggle("active", panel.dataset.panel === name);
  });
  localStorage.setItem("cobot_dashboard_tab", name);
}

function renderJobs(jobs) {
  const list = $("jobsList");
  if (!jobs.length) { list.innerHTML = '<p class="empty">暂无任务</p>'; return; }
  list.innerHTML = "";
  for (const job of jobs) {
    const row = document.createElement("div");
    row.className = "job";
    const name = document.createElement("div");
    name.className = "job-name";
    name.innerHTML = `<strong>${job.kind}</strong><span>${job.id} · PID ${job.pid}</span>`;
    const badge = document.createElement("span");
    badge.className = `badge ${job.state}`;
    badge.textContent = job.state;
    const actions = document.createElement("div");
    const logButton = document.createElement("button");
    logButton.className = "button ghost small";
    logButton.textContent = "Log";
    logButton.onclick = () => selectJob(job);
    actions.appendChild(logButton);
    if (["running", "stopping"].includes(job.state)) {
      const stop = document.createElement("button");
      stop.className = "button ghost small";
      stop.textContent = "Stop";
      stop.onclick = () => stopJob(job.id);
      actions.appendChild(stop);
    }
    row.append(name, badge, actions);
    list.appendChild(row);
  }
}

function renderRlRecordControl(jobs) {
  const anyRecorder = jobs.find((job) => job.kind === "record" && job.state === "running");
  const active = jobs.find((job) =>
    job.kind === "record" && job.state === "running" && job.metadata?.collection_type === "RL"
  ) || null;
  state.activeRlRecord = active;
  $("saveRlSuccessButton").disabled = !active;
  $("saveRlFailureButton").disabled = !active;
  $("abortRlButton").disabled = !active;
  $("startRlButton").disabled = Boolean(anyRecorder);
  $("startRlButton").textContent = "开始 RL 录制";
  $("rlRecordHint").textContent = active
    ? `Recording ${active.id}. ${$("rlTeachAutosave").checked ? "Teach release will auto-save Success." : "Choose Success or Failure, or Abort to discard."}`
    : "加载 IL 模型即采集 RLT 初始数据；加载 Actor 即采集在线数据。采集格式相同，差异只记录在 policy metadata。";
}

function renderIlRecordControl(jobs) {
  const recordKinds = ["record", "record_with_init"];
  const anyRecorder = jobs.find((job) => recordKinds.includes(job.kind) && job.state === "running");
  const active = jobs.find((job) =>
    recordKinds.includes(job.kind) && job.state === "running" && job.metadata?.collection_type === "IL"
  ) || null;
  const preparing = active?.kind === "record_with_init"
    && ["queued", "initializing"].includes(active.metadata?.phase || "queued");
  state.activeIlRecord = active;
  $("startIlButton").disabled = Boolean(anyRecorder);
  $("startIlWithInitButton").disabled = Boolean(anyRecorder);
  $("saveIlSuccessButton").disabled = !active || preparing;
  $("saveIlFailureButton").disabled = !active || preparing;
  $("abortIlButton").disabled = !active;
  $("ilRecordHint").textContent = preparing
    ? `Moving all four arms to Init (${active.id}). Recording will start only after alignment completes.`
    : active
      ? `Recording ${active.id}. Save Success/Failure, or Abort to discard it.`
    : "勾选后 Teach 松开自动保存 Success；Abort 会丢弃本次录制，不保存数据。";
}

function renderPolicyControls(overview) {
  const server = overview.policy_server || { ready: false, state: "offline" };
  state.policyServer = server;
  const selectedPolicy = (overview.policies || []).find((item) => item.id === $("policySelect").value);
  const supportsDenoisingSteps = selectedPolicy?.type === "smolvla";
  const denoisingSelect = $("policyDenoisingSteps");
  $("policyDenoisingControl").hidden = !supportsDenoisingSteps;
  if (server.ready && server.policy_type === "smolvla") {
    denoisingSelect.value = String([5, 10, 20].includes(Number(server.num_steps)) ? server.num_steps : 10);
  }
  denoisingSelect.disabled = !supportsDenoisingSteps || server.ready || server.state === "loading";
  $("policyDenoisingHint").textContent = supportsDenoisingSteps
    ? server.ready || server.state === "loading"
      ? `Loaded setting: ${server.num_steps ?? 10} steps. Stop the Model Server before changing it.`
      : "Controls SmolVLA flow-matching inference steps; the selected value takes effect when Model Server starts."
    : "Denoising-step override is available for SmolVLA checkpoints only.";
  const guidedRtc = server.inference_chunking === "guided_rtc";
  const serverActionSpace = ["joint", "eef_target"].includes(server.action_space)
    ? server.action_space
    : server.relative_eef_actions
      ? "eef_target"
      : "";
  const actionStepsInput = $("rolloutActionSteps");
  const chunkSize = Number(server.chunk_size);
  if (server.ready && Number.isInteger(chunkSize) && chunkSize > 0) {
    const serverKey = [
      server.policy_id || server.policy_path || "policy",
      server.compute || "pc",
      chunkSize,
    ].join("|");
    actionStepsInput.max = String(chunkSize);
    actionStepsInput.placeholder = `All model output (${chunkSize})`;
    if (state.actionHorizonServerKey !== serverKey) {
      actionStepsInput.value = "";
      state.actionHorizonServerKey = serverKey;
    } else if (actionStepsInput.value && Number(actionStepsInput.value) > chunkSize) {
      actionStepsInput.value = String(chunkSize);
    }
  } else {
    actionStepsInput.max = "50";
    actionStepsInput.placeholder = "All model output";
    state.actionHorizonServerKey = null;
  }
  const activeRollout = overview.jobs.find((job) =>
    job.kind === "policy_execute" && ["running", "stopping"].includes(job.state)
  ) || null;
  state.activeRollout = activeRollout;
  const badge = $("policyServerBadge");
  badge.textContent = server.ready ? "Ready" : server.state === "loading" ? "Loading" : "Offline";
  badge.className = `badge ${server.ready ? "online" : server.state === "loading" ? "running" : "offline"}`;
  $("policyServerDetail").textContent = server.ready
    ? `${server.compute === "robot" ? "Robot GPU" : "PC GPU"} · ${server.policy_id || server.policy_path || "model"} · ${guidedRtc ? "Guided RTC terminal" : "Original chunks"}${server.policy_type === "smolvla" ? ` · ${server.num_steps ?? 10} denoise steps` : ""} · ${server.device || "device unknown"}`
    : server.state === "loading"
      ? `Loading checkpoint into ${server.compute === "robot" ? "Robot GPU" : "PC GPU"}…`
      : "No model loaded";
  const selectionLocked = server.ready || server.state === "loading";
  const pendingSelection = policySelectionMismatch(server);
  if (server.compute && selectionLocked) $("policyComputeSelect").value = server.compute;
  $("policyComputeSelect").disabled = selectionLocked;
  if (server.device && selectionLocked) $("policyDeviceSelect").value = server.device;
  $("policyDeviceSelect").disabled = selectionLocked || $("policyComputeSelect").value === "robot";
  if ($("policyComputeSelect").value === "robot") $("policyDeviceSelect").value = "cuda:0";
  const checkpointSelectionLocked = server.state === "loading" || Boolean(activeRollout);
  $("policyRunSelect").disabled = checkpointSelectionLocked;
  $("policySelect").disabled = checkpointSelectionLocked;
  $("policySelectionHint").textContent = pendingSelection
    ? "Selected checkpoint is not loaded. Stop Server, then Start Model Server to load your selection."
    : server.ready ? "Selected checkpoint matches the loaded model." : "Select a checkpoint, then Start Model Server.";
  if (selectionLocked) $("policyInferenceChunking").value = guidedRtc ? "guided_rtc" : "original";
  $("policyInferenceChunking").disabled = selectionLocked;
  $("policyChunkingHint").textContent = guidedRtc
    ? `Synchronous test: execute each native model chunk completely; the next chunk uses the prior terminal target as a one-point RTC prefix. ${server.relative_eef_actions ? "Relative-EEF state/target is selected." : "Absolute joint state/target is selected."} Stop and reload to switch.`
    : "Original sampling: each model chunk is generated independently.";
  if (server.ready && serverActionSpace) {
    $("rolloutActionSpace").value = serverActionSpace;
    $("rolloutActionSpace").disabled = true;
    if (serverActionSpace === "eef_target") {
      $("rolloutEefIkBackend").value = "pink";
      if (!["standard", "eef_v5", "guided_rtc"].includes($("rolloutRunner").value)) {
        $("rolloutRunner").value = "eef_v5";
      }
    }
  }
  if (guidedRtc) {
    $("rolloutRunner").value = "guided_rtc";
    $("rolloutRunner").disabled = true;
    $("rolloutActionSteps").value = "";
    $("rolloutActionSteps").disabled = true;
    $("rolloutActionSpace").value = serverActionSpace || (server.relative_eef_actions ? "eef_target" : "joint");
    $("rolloutActionSpace").disabled = true;
  } else {
    $("rolloutRunner").disabled = false;
    $("rolloutActionSteps").disabled = false;
    if (!server.ready || !serverActionSpace) $("rolloutActionSpace").disabled = false;
  }
  $("startPolicyServerButton").disabled = server.ready || server.state === "loading"
    || overview.policies.length === 0;
  $("stopPolicyServerButton").disabled = !server.ready && server.state !== "loading";
  $("startRolloutButton").disabled = !server.ready || Boolean(activeRollout) || pendingSelection;
  $("startSyncRolloutButton").disabled = !server.ready || Boolean(activeRollout) || pendingSelection;
  $("stopRolloutButton").disabled = !activeRollout;
  $("rolloutHint").textContent = activeRollout
    ? `Mode ${activeRollout.metadata?.mode || "?"} rollout is ${activeRollout.state} on ${activeRollout.metadata?.compute === "robot" ? "Robot" : "PC"}. Teach release will end it.`
    : pendingSelection
      ? "Load the selected checkpoint before starting rollout. The server still holds the previous model."
    : guidedRtc
      ? "Guided RTC: complete model chunks; each next chunk is softly anchored to the previous terminal target. This first test is synchronous."
    : server.ready
      ? `${server.compute === "robot" ? "Robot" : "PC"} runs both inference and rollout. Mode 1 moves puppet only; Mode 2 moves right master + puppet.`
      : "Start Model Server and wait for Ready first.";
}

function renderReplayControls(overview) {
  const activeReplay = overview.jobs.find((job) =>
    job.kind === "replay_execute" && ["running", "stopping"].includes(job.state)
  ) || null;
  state.activeReplay = activeReplay;
  $("startReplayButton").disabled = Boolean(activeReplay);
  $("stopReplayButton").disabled = !activeReplay;
  for (const id of ["replaySource", "replayDataset", "replayEpisode", "replaySpeed", "replayLoop"]) {
    $(id).disabled = Boolean(activeReplay);
  }
  $("replayMotionConfirm").disabled = Boolean(activeReplay);
  const stopSafety = activeReplay?.metadata?.stop_safety;
  $("replayHint").textContent = activeReplay
    ? `Replay ${activeReplay.metadata?.source_type === "lerobot" ? `${activeReplay.metadata.dataset_id} · ep${activeReplay.metadata.episode_index}` : "raw episode"} is ${activeReplay.state}.${stopSafety ? (stopSafety.safe ? " Output-off and SAFE were accepted." : ` STOP WARNING: SAFE was not confirmed (${(stopSafety.errors || []).join("; ")}).`) : " Pause / SAFE disables output and ends replay."} It cannot resume from this frame.`
    : "Pause / SAFE immediately disables output and ends replay. To continue, start the episode again.";
  renderReplaySourceControls();
}

async function refresh() {
  // Manual snapshots only. Repeated clicks share one in-flight request rather
  // than adding more scans to the already slow overview endpoint.
  if (state.refreshPromise) return state.refreshPromise;
  const buttons = [$("refreshButton"), $("refreshJobsButton")].filter(Boolean);
  for (const button of buttons) button.disabled = true;
  const hint = $("stateRefreshHint");
  if (hint) hint.textContent = "正在刷新状态…相机预览保持实时。";
  state.refreshPromise = refreshSnapshot().then((ok) => {
    if (hint) hint.textContent = ok
      ? `目录：${new Date().toLocaleTimeString()}（手动刷新）`
      : "目录刷新失败，请重试";
  }).finally(() => {
    state.refreshPromise = null;
    for (const button of buttons) button.disabled = false;
  });
  return state.refreshPromise;
}

function overlayRuntime(overview) {
  const patch = state.runtimePatch;
  if (patch.jobs) overview.jobs = patch.jobs;
  if (patch.policy_server) overview.policy_server = patch.policy_server;
  if (patch.rlt_policy_server) overview.rlt = { ...overview.rlt, policy_server: patch.rlt_policy_server };
  if (patch.raws.size) {
    const raws = new Map((overview.raws || []).map((raw) => [raw.id, raw]));
    for (const [id, raw] of patch.raws) raws.set(id, raw);
    overview.raws = [...raws.values()].sort((a, b) => b.modified_at_ns - a.modified_at_ns);
  }
  mergeRobotStatus(overview);
}

function mergeRobotStatus(overview) {
  if (!state.robotStatus) return;
  const status = state.robotStatus;
  const old = overview.bridge || {};
  const devices = { ...(old.devices || {}), ...status.devices };
  for (const kind of ["arms", "cameras"]) {
    devices[kind] = { ...(old.devices?.[kind] || {}) };
    for (const [name, flags] of Object.entries(status.devices?.[kind] || {})) {
      devices[kind][name] = { ...devices[kind][name], ...flags };
    }
  }
  // Unknown fields must not inherit a stale ENABLED/Teach/mode indication.
  overview.bridge = { ...old, mode: null, mode_id: null, teach_active: undefined,
    output_enabled: undefined, emergency_latched: undefined, allow_control: false,
    ...status, devices };
}

function updateRobotDisplay() {
  if (!state.overview) {
    renderBridge(state.robotStatus || { online: false });
    return;
  }
  mergeRobotStatus(state.overview);
  renderBridge(state.overview.bridge);
  renderPolicyControls(state.overview);
  renderReplayControls(state.overview);
  renderRltControls(state.overview);
  renderRobotStartup(state.overview);
  renderMotionReadiness(state.overview);
}

function robotStatusStale(reason) {
  clearTimeout(state.robotWatchdog);
  jointStateStale();
  const oldDevices = state.robotStatus?.devices || state.overview?.bridge?.devices || {};
  const devices = { ok: false, ok_count: 0, total: 7, arms: {}, cameras: {} };
  for (const name of Object.keys(oldDevices.arms || {})) devices.arms[name] = { ok: false, fresh: false };
  for (const name of Object.keys(oldDevices.cameras || {})) devices.cameras[name] = { ok: false, fresh: false, has_frame: false };
  state.robotStatus = { online: false, stale: true, devices };
  $("robotStatusHint").textContent = `机器人状态已过期 · ${reason}；控制相关状态暂不可确认。`;
  updateRobotDisplay();
}

function handleRobotEvent(kind, data) {
  if (data.age_ms > 1500) { robotStatusStale("后端检测超时"); return; }
  if (kind === "robot_status_changed") {
    state.robotStatus = data.status;
    state.eventVersion += 1;
    updateRobotDisplay();
  }
  clearTimeout(state.robotWatchdog);
  state.robotWatchdog = setTimeout(() => robotStatusStale("超过3秒未收到状态心跳"), 3000);
  $("robotStatusHint").textContent = `${state.robotStatus?.online ? "机器人在线" : "Bridge 离线"} · ${new Date().toLocaleTimeString()} · 状态切换优先 / 关节 2 Hz`;
}

function handleRuntimeEvent(kind, data) {
  if (kind === "joint_state") { handleJointState(data); return; }
  if (["robot_status_changed", "robot_heartbeat"].includes(kind)) {
    handleRobotEvent(kind, data);
    return; // Do not rebuild directory selectors or scan anything on status changes.
  }
  const patch = state.runtimePatch;
  if (kind === "runtime_snapshot") {
    patch.jobs = data.jobs;
    for (const raw of data.raws || []) patch.raws.set(raw.id, raw);
    if (data.policy_server) patch.policy_server = data.policy_server;
    if (data.rlt_policy_server) patch.rlt_policy_server = data.rlt_policy_server;
  } else if (kind === "job_changed") {
    patch.jobs = data.jobs;
    if (data.raw) patch.raws.set(data.raw.id, data.raw);
  } else if (kind === "server_changed") {
    if (!["policy_server", "rlt_policy_server"].includes(data.name)) return;
    patch[data.name] = data.status;
  } else return;
  state.eventVersion += 1;
  if (state.overview) {
    overlayRuntime(state.overview);
    renderJobs(state.overview.jobs);
    renderIlRecordControl(state.overview.jobs);
    renderRlRecordControl(state.overview.jobs);
    renderPolicyControls(state.overview);
    renderReplayControls(state.overview);
    renderRltControls(state.overview);
    renderRobotStartup(state.overview);
    renderMotionReadiness(state.overview);
    if (data.raw || data.raws?.length) fillRawSelectors(state.overview.raws.filter((raw) => raw.completed));
  }
  if (kind === "job_changed") {
    if (["robot_startup", "robot_self_check"].includes(data.job.kind) && data.job.state === "completed") {
      toast(data.job.kind === "robot_startup" ? "机器人服务已就绪，运动输出保持关闭。" : "自检完成，请查看报告。");
    }
    if (["record", "record_with_init"].includes(data.job.kind) && data.raw?.completed) {
      const outcome = data.raw.outcome === "success" ? "成功" : data.raw.outcome === "failure" ? "失败" : data.raw.outcome;
      const message = `录制已保存 · ${outcome} · ${data.raw.episode}`;
      if (data.raw.collection_type === "IL" && !state.activeIlRecord) $("ilRecordHint").textContent = message;
      toast(message);
    } else if (data.job.state === "failed") {
      toast(`${data.job.kind} 已停止或失败，请查看任务日志。`, true);
    }
    if (state.selectedJob === data.job.id) refreshLog();
  }
}

async function startRuntimeEvents() {
  const hint = $("runtimeEventHint");
  state.robotWatchdog = setTimeout(() => robotStatusStale("尚未收到机器人状态"), 3000);
  while (!state.eventsStopped) {
    state.eventController = new AbortController();
    try {
      const requestHeaders = headers();
      if (state.eventCursor !== null) requestHeaders["Last-Event-ID"] = state.eventCursor;
      const response = await fetch("/api/events", { headers: requestHeaders, signal: state.eventController.signal });
      if (!response.ok || !response.body) throw new Error(`HTTP ${response.status}`);
      if (hint) hint.textContent = "SSE 已连接（状态 / 任务 / 模型自动更新）";
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      try {
        while (!state.eventsStopped) {
          const { value, done } = await reader.read();
          if (done) throw new Error("通知连接已断开");
          buffer += decoder.decode(value, { stream: true });
          if (buffer.length > 2 * 1024 * 1024) throw new Error("通知消息过大");
          let end;
          while ((end = buffer.indexOf("\n\n")) >= 0) {
            const frame = buffer.slice(0, end); buffer = buffer.slice(end + 2);
            let id = null, kind = "message"; const lines = [];
            for (const line of frame.split("\n")) {
              if (line.startsWith("id:")) id = line.slice(3).trim();
              if (line.startsWith("event:")) kind = line.slice(6).trim();
              if (line.startsWith("data:")) lines.push(line.slice(5).trimStart());
            }
            if (lines.length) {
              handleRuntimeEvent(kind, JSON.parse(lines.join("\n")));
              if (id !== null) state.eventCursor = id;
            }
          }
        }
      } finally { await reader.cancel().catch(() => {}); }
    } catch (error) {
      if (state.eventsStopped) break;
      robotStatusStale("SSE 连接断开");
      if (hint) hint.textContent = "SSE 断线，正在重连";
    }
    if (!state.eventsStopped) await new Promise((resolve) => setTimeout(resolve, 3000));
  }
}

async function refreshSnapshot() {
  try {
    const eventVersion = state.eventVersion;
    const liveNow = LIVE_CAMERA_NAMES.filter((name) => state.preview[name]).join(",");
    const overview = await api(liveNow ? `/api/overview?live_cameras=${encodeURIComponent(liveNow)}` : "/api/overview");
    if (!state.overview || state.eventVersion !== eventVersion) overlayRuntime(overview);
    else mergeRobotStatus(overview); // Slow overview must never replace live control status.
    state.overview = overview;
    renderBridge(overview.bridge);
    renderCapabilities(overview.capabilities);
    if (state.selectedJob && !overview.jobs.some((job) => job.id === state.selectedJob)) {
      state.selectedJob = null;
      $("logTitle").textContent = "选择一个任务查看日志";
      $("logOutput").textContent = "No job selected.";
    }
    renderJobs(overview.jobs);
    renderIlRecordControl(overview.jobs);
    renderRlRecordControl(overview.jobs);
    fillRawSelectors(overview.raws.filter((item) => item.completed));
    fillLerobotReplayDatasets(overview.lerobot_replay_datasets || []);
    renderReplayControls(overview);
    fillPolicySelectors(overview.policies, overview.policy_server?.policy_id || "");
    renderPolicyControls(overview);
    fillRltCheckpointSelector(
      overview.rlt?.stage1_checkpoints || [],
      overview.rlt?.policy_server?.checkpoint_id || "",
    );
    renderRltControls(overview);
    renderRobotStartup(overview);
    $("rlAutoStartPolicy").disabled = false;
    renderMotionReadiness(overview);
    $("dataRootHint").textContent = `Raw: ${overview.paths.raw_root} · Datasets: ${overview.paths.dataset_root}`;
    if (state.selectedJob) await refreshLog();
    return true;
  } catch (error) {
    if (state.robotStatus) updateRobotDisplay();
    else renderBridge({ online: false });
    toast(error.message, true);
    return false;
  }
}

function renderMotionReadiness(overview) {
    const motionKinds = [
      "replay_execute", "policy_execute", "rlt_policy_execute", "rlt_online_execute", "init_pose_execute"
    ];
    const initRecordBusy = overview.jobs.some((job) =>
      job.kind === "record_with_init"
      && ["running", "stopping"].includes(job.state)
      && ["queued", "initializing"].includes(job.metadata?.phase || "queued")
    );
    const motionBusy = overview.jobs.some((job) =>
      motionKinds.includes(job.kind) && ["running", "stopping"].includes(job.state)
    ) || initRecordBusy;
    const restartBusy = overview.jobs.some((job) =>
      ["device_restart", "robot_startup", "robot_self_check"].includes(job.kind) && ["running", "stopping"].includes(job.state)
    );
    const recordBusy = overview.jobs.some((job) =>
      ["record", "record_with_init"].includes(job.kind)
      && ["running", "stopping"].includes(job.state)
    );
    $("restartDevicesButton").disabled = restartBusy || recordBusy || overview.bridge.teach_active;
    $("restartDevicesButton").textContent = restartBusy ? "Restarting…" : "Restart All Devices";
    $("restartDevicesHint").textContent = restartBusy
      ? "Waiting for four fresh arm states and three fresh camera frames…"
      : recordBusy
        ? "Finish or abort the active recording before restarting devices."
        : overview.bridge.teach_active
          ? "Release Teach before restarting devices."
          : "Stops rollout, then restarts four-arm ROS and all three camera nodes. Bridge and model server stay online.";
    $("exitStopButton").disabled = !overview.bridge.online || !overview.bridge.emergency_latched;
    const initPoseDisabled = !overview.bridge.online
      || !overview.bridge.allow_control
      || !overview.bridge.devices?.ok
      || overview.bridge.teach_active
      || overview.bridge.emergency_latched
      || motionBusy
      || recordBusy;
    $("moveInitPoseButton").disabled = initPoseDisabled;
    $("moveRlInitPoseButton").disabled = initPoseDisabled;
    $("moveRltInitPoseButton").disabled = initPoseDisabled;
    $("startIlWithInitButton").disabled = initPoseDisabled || recordBusy;
    $("startIlWithInitButton").textContent = initRecordBusy
      ? "Initializing…"
      : "Init → 开始 IL 录制";
}

async function selectJob(job) {
  state.selectedJob = job.id;
  $("logTitle").textContent = `${job.kind} · ${job.id}`;
  await refreshLog();
}

async function refreshLog() {
  if (!state.selectedJob) return;
  try {
    const response = await fetch(`/api/jobs/${state.selectedJob}/log`, { headers: headers() });
    $("logOutput").textContent = await response.text();
    $("logOutput").scrollTop = $("logOutput").scrollHeight;
  } catch (error) { $("logOutput").textContent = error.message; }
}

async function stopJob(id) {
  try { await api(`/api/jobs/${id}`, { method: "DELETE" }); toast(`Stopping ${id}`); await refresh(); }
  catch (error) { toast(error.message, true); }
}

function recordPayload(prefix, policyId) {
  const isIl = prefix === "il";
  return {
    collection: $(isIl ? "ilCollection" : "rlCollection").value,
    task: $(isIl ? "ilTask" : "rlTask").value,
    side: "right",
    cameras: isIl && !$("ilRightCamera").checked ? ["front"] : ["front", "right"],
    fps: Number(isIl ? $("ilFps").value : $("rlFps").value),
    max_seconds: 3600,
    stop_on_teach_release: $(isIl ? "ilTeachAutosave" : "rlTeachAutosave").checked,
    policy_id: policyId,
    collection_type: isIl ? "IL" : "RL",
  };
}

async function startIl() {
  try {
    const result = await api("/api/jobs/record", { method: "POST", body: JSON.stringify(recordPayload("il", "human-il")) });
    toast("IL recorder started. Press Teach to demonstrate, then release to finish.");
    state.selectedJob = result.id;
    await refresh();
  } catch (error) { toast(error.message, true); }
}

async function startIlWithInit() {
  try {
    const payload = { ...recordPayload("il", "human-il"), confirm_motion: true };
    const result = await api("/api/jobs/init-and-record", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    state.selectedJob = result.id;
    toast("Moving to Init. IL recording will start automatically after all four arms arrive.");
    await refresh();
  } catch (error) { toast(error.message, true); }
}

async function finalizeIl(outcome) {
  const job = state.activeIlRecord;
  if (!job) return toast("No active IL recorder", true);
  const label = outcome === "success" ? "SUCCESS" : "FAILURE";
  if (!window.confirm(`Finalize and save this IL episode as ${label}?`)) return;
  try {
    await api(`/api/jobs/${job.id}/finalize-record`, {
      method: "POST",
      body: JSON.stringify({ outcome }),
    });
    state.activeIlRecord = null;
    $("saveIlSuccessButton").disabled = true;
    $("saveIlFailureButton").disabled = true;
    toast(`Saving IL episode as ${label}.`);
  } catch (error) { toast(error.message, true); }
}

async function abortRecord(kind) {
  const isIl = kind === "IL";
  const job = isIl ? state.activeIlRecord : state.activeRlRecord;
  if (!job) return toast(`No active ${kind} recorder`, true);
  const motionNote = isIl ? "" : " The active rollout will also stop.";
  if (!window.confirm(`Abort this ${kind} recording and permanently discard its data?${motionNote}`)) return;
  try {
    await api(`/api/jobs/${job.id}/abort-record`, { method: "POST" });
    if (isIl) {
      state.activeIlRecord = null;
      $("saveIlSuccessButton").disabled = true;
      $("saveIlFailureButton").disabled = true;
      $("abortIlButton").disabled = true;
    } else {
      state.activeRlRecord = null;
      $("saveRlSuccessButton").disabled = true;
      $("saveRlFailureButton").disabled = true;
      $("abortRlButton").disabled = true;
    }
    toast(`${kind} recording aborted. Its raw data is being discarded.`);
  } catch (error) { toast(error.message, true); }
}

async function startRl() {
  const autoStartPolicy = $("rlAutoStartPolicy").checked;
  const policyMode = Number($("rlPolicyMode").value);
  const policyTarget = policyMode === 1 ? "right puppet only" : "right master + puppet";
  let recorder = null;
  try {
    if (autoStartPolicy) {
      validatePolicySelection();
      if (!state.policyServer?.ready) {
        throw new Error("Start Model Server and wait for Ready before auto-starting rollout");
      }
      if (policySelectionMismatch(state.policyServer)) {
        throw new Error("Load the selected checkpoint before auto-starting rollout");
      }
      if (!$("rolloutMotionConfirm").checked) {
        throw new Error("Enable the real-motion confirmation in Policy Rollout first");
      }
      if (![1, 2].includes(policyMode)) throw new Error(`Invalid RL rollout mode: ${policyMode}`);
      if (!window.confirm(`Start RL recording and run Mode ${policyMode} on the real ${policyTarget}?`)) return;
    }
    const selectedPolicy = $("policySelect").value || "rl-online-collection";
    recorder = await api("/api/jobs/record", {
      method: "POST",
      body: JSON.stringify(recordPayload("rl", selectedPolicy)),
    });
    state.selectedJob = recorder.id;
    if (autoStartPolicy) {
      try {
        const policy = await launchPolicy(policyMode);
        state.selectedJob = policy.id;
      } catch (error) {
        await api(`/api/jobs/${recorder.id}`, { method: "DELETE" }).catch(() => null);
        throw new Error(`Policy did not start; recorder is stopping: ${error.message}`);
      }
      toast(`RL recorder and Mode ${policyMode} policy started together on ${policyTarget}.`);
    } else {
      toast("RL recorder started. Start rollout, then save Success or Failure.");
    }
    await refresh();
  } catch (error) { toast(error.message, true); }
}

async function finalizeRl(outcome) {
  const job = state.activeRlRecord;
  if (!job) return toast("No active RL recorder", true);
  const label = outcome === "success" ? "SUCCESS" : "FAILURE";
  if (!window.confirm(`Finalize and save this RL episode as ${label}? The active rollout will stop.`)) return;
  try {
    await api(`/api/jobs/${job.id}/finalize-record`, {
      method: "POST",
      body: JSON.stringify({ outcome }),
    });
    state.activeRlRecord = null;
    $("saveRlSuccessButton").disabled = true;
    $("saveRlFailureButton").disabled = true;
    toast(`Saving RL episode as ${label}. Rollout is stopping.`);
  } catch (error) { toast(error.message, true); }
}

async function startReplay() {
  if (!$("replayMotionConfirm").checked) return toast("请先确认 replay 真实运动安全条件", true);
  if (!window.confirm("Replay recorded actions on the real robot?")) return;
  try {
    let result;
    if ($("replaySource").value === "lerobot") {
      const dataset = (state.overview?.lerobot_replay_datasets || [])
        .find((item) => item.id === $("replayDataset").value);
      const episodeIndex = Number($("replayEpisode").value);
      if (!dataset) throw new Error("Select a LeRobot dataset first");
      if (!Number.isInteger(episodeIndex) || episodeIndex < 0 || episodeIndex >= dataset.episodes) {
        throw new Error(`Episode index must be between 0 and ${dataset.episodes - 1}`);
      }
      result = await api("/api/jobs/replay-lerobot", {
        method: "POST",
        body: JSON.stringify({
          dataset_id: dataset.id,
          episode_index: episodeIndex,
          side: "right",
          waypoint_hz: 30,
          speed_scale: Number($("replaySpeed").value),
          alignment_speed: 0.05,
          alignment_min_seconds: 5,
          execute: true,
          confirm_motion: true,
        }),
      });
    } else {
      if (!$("convertRaw").value) throw new Error("No raw episode selected in Data browser");
      result = await api("/api/jobs/replay", {
        method: "POST",
        body: JSON.stringify({
          raw_id: $("convertRaw").value, side: "right", source: "puppet_command",
          mode: "sync", waypoint_hz: 30, speed_scale: Number($("replaySpeed").value),
          loop: $("replayLoop").checked, execute: true, confirm_motion: true,
        }),
      });
    }
    state.selectedJob = result.id;
    toast("Replay started. Use Pause / SAFE whenever you want to stop at the current point.");
    await refresh();
  } catch (error) { toast(error.message, true); }
}

async function stopReplay() {
  if (!state.activeReplay) return toast("No active replay", true);
  try {
    const result = await api(`/api/jobs/${state.activeReplay.id}`, { method: "DELETE" });
    const safety = result.metadata?.stop_safety;
    if (safety && !safety.safe) {
      toast(`Replay stopped, but SAFE was not confirmed: ${(safety.errors || []).join("; ")}`, true);
    } else if (safety?.safe) {
      toast("Pause / SAFE accepted: output disabled and replay is stopping.");
    } else {
      toast("Replay stop requested, but no SAFE acknowledgement was returned. Check System output status.", true);
    }
    await refresh();
  } catch (error) { toast(error.message, true); }
}

function validatePolicySelection() {
  if (!$("policySelect").value) {
    throw new Error("Policy checkpoint is required");
  }
}

async function launchPolicy(mode = 2) {
  if (policySelectionMismatch(state.policyServer)) {
    throw new Error("Selected checkpoint is not loaded. Stop Server, then Start Model Server first.");
  }
  const actionStepsValue = $("rolloutActionSteps").value.trim();
  const actionSpace = $("rolloutActionSpace").value;
  if (["eef_delta", "eef_target"].includes(actionSpace) && mode === 2 && $("rolloutEefIkBackend").value !== "pink") {
    throw new Error("Mode 2 EEF rollout requires PC Pink IK to convert EEF targets into synchronized joint chunks");
  }
  const runner = $("rolloutRunner").value;
  if (["eef_delta", "eef_target"].includes(actionSpace) && !["standard", "eef_v5", "guided_rtc"].includes(runner)) {
    throw new Error("EEF action spaces require Standard, EEF V5, or Guided RTC runner");
  }
  if (runner === "guided_rtc") {
    if (state.policyServer?.inference_chunking !== "guided_rtc") {
      throw new Error("Stop and reload the Model Server with Guided RTC first");
    }
    if (actionSpace === "eef_delta") {
      throw new Error("Guided RTC needs absolute joint or EEF-target actions");
    }
    if (actionStepsValue) {
      throw new Error("Guided RTC always executes the complete model chunk; clear Actions consumed");
    }
  }
  return api("/api/jobs/policy", {
    method: "POST",
    body: JSON.stringify({
      task: $("rlTask").value,
      side: "right", fps: Number($("rolloutFps").value), duration: 3600,
      mode, action_space: actionSpace, runner,
      eef_ik_backend: $("rolloutEefIkBackend").value,
      n_action_steps: actionSpace === "eef_delta" ? 1 : (actionStepsValue ? Number(actionStepsValue) : null),
      execute: true, confirm_motion: true,
    }),
  });
}

async function startPolicyServer() {
  try {
    validatePolicySelection();
    const selectedPolicy = (state.overview?.policies || []).find((item) => item.id === $("policySelect").value);
    const result = await api("/api/policy-server/start", {
      method: "POST",
      body: JSON.stringify({
        policy_id: $("policySelect").value,
        device: $("policyDeviceSelect").value,
        compute: $("policyComputeSelect").value,
        inference_chunking: $("policyInferenceChunking").value,
        num_steps: selectedPolicy?.type === "smolvla" ? Number($("policyDenoisingSteps").value) : null,
      }),
    });
    if (result.id) state.selectedJob = result.id;
    const target = $("policyComputeSelect").value === "robot" ? "Robot GPU" : "PC GPU";
    toast(result.already_running ? `Selected model is already Ready on ${target}.` : `Loading model into ${target}. Wait for Ready.`);
    await refresh();
  } catch (error) { toast(error.message, true); }
}

async function stopPolicyServer() {
  if (!window.confirm("Stop the model server and unload the model from GPU?")) return;
  try {
    await api("/api/policy-server/stop", { method: "POST" });
    toast("Model server is stopping. This does not move the robot.");
    await refresh();
  } catch (error) { toast(error.message, true); }
}

async function startRollout(mode) {
  if (!state.policyServer?.ready) return toast("Model Server is not Ready", true);
  if (policySelectionMismatch(state.policyServer)) return toast("Load the selected checkpoint first", true);
  if (!$("rolloutMotionConfirm").checked) return toast("请先确认 rollout 真实运动安全条件", true);
  const target = mode === 2 ? "right master + puppet with individual alignment" : "right puppet only";
  if (!window.confirm(`Run Mode ${mode} policy on ${target}?`)) return;
  try {
    const result = await launchPolicy(mode);
    state.selectedJob = result.id;
    toast(`Mode ${mode} policy rollout started.`);
    await refresh();
  } catch (error) { toast(error.message, true); }
}

async function stopRollout() {
  if (!state.activeRollout) return toast("No active rollout", true);
  try {
    await api(`/api/jobs/${state.activeRollout.id}`, { method: "DELETE" });
    toast("Rollout is stopping; model server stays loaded.");
    await refresh();
  } catch (error) { toast(error.message, true); }
}

async function validateRlt() {
  const checkpointId = $("rltCheckpointSelect").value;
  if (!checkpointId) return toast("Select an RLT Stage 1 checkpoint", true);
  try {
    const result = await api("/api/jobs/rlt-validate", {
      method: "POST",
      body: JSON.stringify({ checkpoint_id: checkpointId }),
    });
    state.selectedJob = result.id;
    toast("Stage 1 strict validation started; no robot connection or motion.");
    await refresh();
  } catch (error) { toast(error.message, true); }
}

async function startRltShadow() {
  const checkpointId = $("rltCheckpointSelect").value;
  if (!checkpointId) return toast("Select an RLT Stage 1 checkpoint", true);
  try {
    const result = await api("/api/jobs/rlt-shadow", {
      method: "POST",
      body: JSON.stringify({
        checkpoint_id: checkpointId,
        task: $("rltTask").value,
        side: "right",
        fps: Number($("rltShadowFps").value),
        duration: Number($("rltShadowDuration").value),
        device: "cuda:0",
      }),
    });
    state.selectedJob = result.id;
    toast("RLT shadow started on PC. It cannot send robot actions.");
    await refresh();
  } catch (error) { toast(error.message, true); }
}

async function stopRltShadow() {
  if (!state.activeRltShadow) return toast("No active RLT shadow", true);
  try {
    await api(`/api/jobs/${state.activeRltShadow.id}`, { method: "DELETE" });
    toast("RLT shadow is stopping; no robot action was sent.");
    await refresh();
  } catch (error) { toast(error.message, true); }
}

async function startRltServer() {
  const checkpointId = $("rltCheckpointSelect").value;
  const actorCheckpointId = $("rltActorCheckpointSelect").value;
  if (!checkpointId) return toast("Select an RLT Stage 1 checkpoint", true);
  if (!actorCheckpointId) return toast("Select an RLT Actor checkpoint", true);
  try {
    const result = await api("/api/rlt-policy-server/start", {
      method: "POST",
      body: JSON.stringify({
        checkpoint_id: checkpointId,
        actor_checkpoint_id: actorCheckpointId,
        device: "cuda:0",
      }),
    });
    if (result.id) state.selectedJob = result.id;
    toast(result.already_running
      ? "RLT rollout server is already loaded."
      : "RLT rollout server is loading on GPU 0; it will stay resident between episodes.");
    await refresh();
  } catch (error) { toast(error.message, true); }
}

async function stopRltServer() {
  try {
    await api("/api/rlt-policy-server/stop", { method: "POST" });
    toast("RLT rollout server is stopping. No robot command was sent.");
    await refresh();
  } catch (error) { toast(error.message, true); }
}

async function rollbackRlt30k() {
  const checkpointId = $("rltCheckpointSelect").value;
  if (!checkpointId) return toast("Select an RLT Stage 1 checkpoint", true);
  if (!window.confirm("Rollback the active Stage 2 Actor to the validated 30k checkpoint? No robot motion will be sent.")) return;
  try {
    const result = await api("/api/rlt-stage2/rollback-30k", {
      method: "POST",
      body: JSON.stringify({ checkpoint_id: checkpointId, target_updates: 30000 }),
    });
    toast(result.already_active
      ? "The 30k Actor is already active."
      : `Actor rolled back from ${result.previous_updates || "unknown"} to 30000 updates.`);
    await refresh();
  } catch (error) { toast(error.message, true); }
}

async function startRltOnline(workflowMode) {
  const checkpointId = $("rltCheckpointSelect").value;
  if (!checkpointId) return toast("Select an RLT Stage 1 checkpoint", true);
  if (!$("rltMotionConfirm").checked) {
    return toast("Confirm the real Online RLT motion safety checkbox first", true);
  }
  const modeName = workflowMode === 1 ? "Actor Mode 1 autonomous" : "Actor Mode 2 human intervention";
  if (!window.confirm(`Start ${modeName} on the real right arm?`)) return;
  try {
    const result = await api("/api/jobs/rlt-online", {
      method: "POST",
      body: JSON.stringify({
        checkpoint_id: checkpointId,
        task: $("rltTask").value,
        workflow_mode: workflowMode,
        rollout_phase: "actor",
        collection: $("rltOnlineCollection").value,
        raw_fps: 30,
        chunk_hz: 30,
        rollout_speed: Number($("rltRolloutSpeed").value),
        smooth_profile: $("rltSmoothProfile").value,
        actor_eef_gate: $("rltActorEefGate").checked,
        actor_max_decisions: Number($("rltActorDecisionBudget").value),
        max_seconds: 3600,
        actor_exploration_std: Number($("rltExplorationStd").value),
        q_select_k: Number($("rltQSelect")?.value || 0),
        execute: true,
        confirm_motion: true,
      }),
    });
    state.selectedJob = result.id;
    toast(`${modeName} started with trained actor enabled=${result.actor_enabled}. `
      + "The arm holds still for 10-15 s while the first chunk is planned.");
    await refresh();
  } catch (error) { toast(error.message, true); }
}

async function startRltReference() {
  const checkpointId = $("rltCheckpointSelect").value;
  if (!checkpointId) return toast("Select an RLT Stage 1 checkpoint", true);
  if (!$("rltMotionConfirm").checked) {
    return toast("Confirm the real reference-only motion safety checkbox first", true);
  }
  if (!window.confirm("Start the Stage 1 reference VLA on the real right puppet? Actor residual and exploration are disabled.")) return;
  try {
    const result = await api("/api/jobs/rlt-online", {
      method: "POST",
      body: JSON.stringify({
        checkpoint_id: checkpointId,
        task: $("rltTask").value,
        workflow_mode: 1,
        rollout_phase: "reference",
        collection: $("rltOnlineCollection").value,
        raw_fps: 30,
        chunk_hz: 30,
        rollout_speed: Number($("rltRolloutSpeed").value),
        smooth_profile: $("rltSmoothProfile").value,
        actor_eef_gate: $("rltActorEefGate").checked,
        max_seconds: 3600,
        actor_exploration_std: 0,
        execute: true,
        confirm_motion: true,
      }),
    });
    state.selectedJob = result.id;
    toast("Reference-only Mode 1 started; Actor and asynchronous training are disabled.");
    await refresh();
  } catch (error) { toast(error.message, true); }
}

async function finalizeRlt(outcome) {
  const job = state.activeRltOnline;
  if (!job) return toast("No active Online RLT episode", true);
  const label = outcome === "success" ? "SUCCESS · Reward 1" : "FAILURE · Reward 0";
  if (!window.confirm(`Finish this Online RLT episode as ${label}? Robot output will stop immediately.`)) return;
  try {
    await api(`/api/jobs/${job.id}/finalize-rlt`, {
      method: "POST",
      body: JSON.stringify({ outcome }),
    });
    toast(`Actor RLT result saved as ${label}. Replay, training and archive are queued.`);
    await refresh();
  } catch (error) { toast(error.message, true); }
}

async function abortRlt() {
  const job = state.activeRltOnline;
  if (!job) return toast("No active Online RLT episode", true);
  if (!window.confirm("Abort Online RLT, stop robot output and discard raw/decision learning data?")) return;
  try {
    await api(`/api/jobs/${job.id}/abort-rlt`, { method: "POST" });
    toast("Online RLT aborted. Robot output is off and the episode will not enter replay.");
    await refresh();
  } catch (error) { toast(error.message, true); }
}

async function markInterventionPhase(phase, quiet = false) {
  const job = state.activeRltOnline;
  if (!job) {
    if (!quiet) toast("No active Online RLT episode", true);
    return;
  }
  try {
    await api(`/api/jobs/${job.id}/intervention-phase`, { method: "POST", body: JSON.stringify({ phase }) });
    if (!quiet) $("interventionPhaseHint").textContent = `已标记 ${phase} @ ${new Date().toLocaleTimeString()}；该时刻之后的 Teach 段按此类型入库。`;
  } catch (error) {
    if (!quiet) toast(error.message, true);
    else console.warn("intervention-phase mark failed:", error);
  }
}
for (const id of ["phaseCalibrateButton", "phaseRescueButton", "phaseRetreatButton"]) {
  $(id).onchange = () => {
    const selected = document.querySelector('input[name="interventionPhase"]:checked');
    if (selected) markInterventionPhase(selected.value);
  };
}

async function loadTeachSegments() {
  const job = state.latestRltOnline;
  if (!job) return toast("没有可标注的 Online RLT episode", true);
  try {
    const result = await api(`/api/jobs/${job.id}/teach-segments`);
    const list = $("teachSegmentList");
    list.innerHTML = "";
    if (!result.segments.length) {
      list.innerHTML = '<p class="hint">本 episode 未检测到 Teach 介入段（全程自主），无需补标。</p>';
    }
    for (const seg of result.segments) {
      const row = document.createElement("div");
      row.className = "hint";
      row.innerHTML = `Teach 段 ${seg.index + 1} · t+${seg.start_s}s · ${seg.dur_s}s　` +
        [["calibrate", "校准"], ["rescue", "救援"], ["retreat", "撤离"]].map(([value, label]) =>
          `<label><input type="radio" name="teach_seg_${seg.index}" value="${value}"${value === "calibrate" ? " checked" : ""}> ${label}</label>`
        ).join(" ");
      list.appendChild(row);
    }
    $("saveTeachLabelsButton").disabled = !result.segments.length;
    $("teachSegmentHint").textContent = `检测到 ${result.segments.length} 个 Teach 段；选好后点“保存补标”，可反复修改，以最后一次为准。`;
  } catch (error) { toast(error.message, true); }
}

async function saveTeachLabels() {
  const job = state.latestRltOnline;
  if (!job) return;
  const labels = [...$("teachSegmentList").querySelectorAll('input[type="radio"]:checked')].map((input) => ({
    index: Number(input.name.replace("teach_seg_", "")),
    phase: input.value,
  }));
  if (!labels.length) return toast("没有可保存的标注", true);
  try {
    const result = await api(`/api/jobs/${job.id}/teach-labels`, { method: "POST", body: JSON.stringify({ labels }) });
    $("teachSegmentHint").textContent = `已保存 ${result.labeled}/${result.segments} 段标注；下次训练加载 replay 时自动生效。`;
    toast("Teach 段补标已保存。");
  } catch (error) { toast(error.message, true); }
}
$("loadTeachSegmentsButton").onclick = loadTeachSegments;
$("saveTeachLabelsButton").onclick = saveTeachLabels;

async function convertRaw() {
  try {
    if (!$("convertRaw").value) throw new Error("No raw episode selected");
    const result = await api("/api/jobs/convert", {
      method: "POST",
      body: JSON.stringify({
        raw_id: $("convertRaw").value,
        dataset_name: $("datasetName").value,
        action_source: $("convertActionSource").value,
        trim_stationary_holds: $("convertTrimHolds").checked,
        use_videos: true,
      }),
    });
    state.selectedJob = result.id;
    toast("Conversion started");
    await refresh();
  } catch (error) { toast(error.message, true); }
}

async function emergencySafe() {
  try { await api("/api/emergency-safe", { method: "POST" }); toast("Stop latched for policy. Physical Teach remains highest priority."); }
  catch (error) { toast(error.message, true); }
}

async function clearEmergency() {
  try { await api("/api/clear-emergency", { method: "POST" }); toast("Stop latch cleared; mode remains SAFE and policy output remains off."); }
  catch (error) { toast(error.message, true); }
}

async function restartDevices() {
  try {
    const result = await api("/api/system/restart-devices", { method: "POST" });
    state.selectedJob = result.id;
    toast("Restarting four arms and all three cameras. Waiting for 7/7 OK.");
    await refresh();
  } catch (error) { toast(error.message, true); }
}

function renderRobotStartup(overview) {
  const active = (job) => ["running", "stopping"].includes(job.state);
  const busy = overview.jobs.some((job) => active(job) && [
    "robot_startup", "robot_self_check", "device_restart", "record", "record_with_init",
    "policy_execute", "replay_execute", "rlt_policy_execute", "rlt_online_execute", "init_pose_execute",
  ].includes(job.kind));
  $("startRobotButton").disabled = busy || overview.bridge?.teach_active || overview.bridge?.output_enabled;
  $("checkRobotButton").disabled = busy;
  const job = overview.jobs.find((job) => ["robot_startup", "robot_self_check"].includes(job.kind));
  if (!job) return;
  const report = job.metadata?.startup_report || {};
  const phases = {queued: "排队中", self_check: "自检", plan: "生成补启动计划", start_bridge: "启动 bridge",
    configure_can: "配置 CAN", start_arms: "启动四臂驱动", start_cameras: "启动相机",
    start_preview: "启动本地预览", verify: "复检并等待数据持续更新", ready: "服务就绪（输出关闭）",
    checked: "只读自检完成", failed: "启动未完成"};
  $("robotStartupHint").textContent = `${phases[report.phase] || job.state} · ${report.error || report.message || "进度和详细原因见报告 / Jobs & Logs"}`;
  $("robotStartupReport").textContent = JSON.stringify(report, null, 2);
  $("restartDevicesButton").disabled = busy || overview.bridge?.teach_active;
}

async function startRobot(checkOnly = false) {
  $("startRobotButton").disabled = true;
  $("checkRobotButton").disabled = true;
  try {
    const result = await api("/api/system/start-robot", {
      method: "POST", body: JSON.stringify({ check_only: checkOnly }),
    });
    state.selectedJob = result.id;
    toast(checkOnly ? "开始只读自检" : "开始自检并补启动缺失服务；不会移动机械臂");
    await refresh();
  } catch (error) {
    toast(error.message, true);
    if (state.overview) renderRobotStartup(state.overview);
  }
}

async function moveInitPose() {
  try {
    const result = await api("/api/jobs/init-pose", {
      method: "POST",
      body: JSON.stringify({ execute: true, confirm_motion: true }),
    });
    state.selectedJob = result.id;
    toast("Four-arm ready-pose motion started. Teach or EMERGENCY SAFE can interrupt it.");
    await refresh();
  } catch (error) { toast(error.message, true); }
}

$("startIlButton").onclick = startIl;
$("startIlWithInitButton").onclick = startIlWithInit;
$("saveIlSuccessButton").onclick = () => finalizeIl("success");
$("saveIlFailureButton").onclick = () => finalizeIl("failure");
$("abortIlButton").onclick = () => abortRecord("IL");
$("startRlButton").onclick = startRl;
$("saveRlSuccessButton").onclick = () => finalizeRl("success");
$("saveRlFailureButton").onclick = () => finalizeRl("failure");
$("abortRlButton").onclick = () => abortRecord("RL");
$("startReplayButton").onclick = startReplay;
$("stopReplayButton").onclick = stopReplay;
$("startRolloutButton").onclick = () => startRollout(1);
$("startSyncRolloutButton").onclick = () => startRollout(2);
$("stopRolloutButton").onclick = stopRollout;
$("startPolicyServerButton").onclick = startPolicyServer;
$("stopPolicyServerButton").onclick = stopPolicyServer;
$("validateRltButton").onclick = validateRlt;
$("startRltShadowButton").onclick = startRltShadow;
$("stopRltShadowButton").onclick = stopRltShadow;
$("startRltServerButton").onclick = startRltServer;
$("stopRltServerButton").onclick = stopRltServer;
$("rollbackRlt30kButton").onclick = rollbackRlt30k;
$("rltCheckpointSelect").onchange = () => renderRltControls(state.overview || { jobs: [], bridge: {} });
$("rltActorCheckpointSelect").onchange = () => renderRltControls(state.overview || { jobs: [], bridge: {} });
$("policyRunSelect").onchange = () => {
  const runId = $("policyRunSelect").value;
  localStorage.setItem(POLICY_RUN_STORAGE_KEY, runId);
  fillCheckpointSelect(state.overview?.policies || [], runId);
  localStorage.setItem(POLICY_CHECKPOINT_STORAGE_KEY, $("policySelect").value);
  if (state.overview) renderPolicyControls(state.overview);
};
$("policySelect").onchange = () => {
  localStorage.setItem(POLICY_RUN_STORAGE_KEY, $("policyRunSelect").value);
  localStorage.setItem(POLICY_CHECKPOINT_STORAGE_KEY, $("policySelect").value);
  if (state.overview) renderPolicyControls(state.overview);
};
$("policyComputeSelect").onchange = () => {
  const robot = $("policyComputeSelect").value === "robot";
  if (robot) $("policyDeviceSelect").value = "cuda:0";
  $("policyDeviceSelect").disabled = robot;
};
const savedDenoisingSteps = localStorage.getItem(POLICY_DENOISING_STEPS_STORAGE_KEY);
if (["5", "10", "20"].includes(savedDenoisingSteps)) {
  $("policyDenoisingSteps").value = savedDenoisingSteps;
}
$("policyDenoisingSteps").onchange = () => {
  localStorage.setItem(POLICY_DENOISING_STEPS_STORAGE_KEY, $("policyDenoisingSteps").value);
};
$("policyInferenceChunking").onchange = () => {
  if ($("policyInferenceChunking").value === "guided_rtc") {
    $("rolloutRunner").value = "guided_rtc";
    $("rolloutActionSteps").value = "";
    $("rolloutFps").value = "30";
  }
};
$("convertRawCollection").onchange = () => {
  renderRawEpisodeList(
    (state.overview?.raws || []).filter((item) => item.completed),
    $("convertRawCollection").value,
  );
};
$("replaySource").onchange = renderReplaySourceControls;
$("replayDataset").onchange = () => {
  localStorage.setItem(REPLAY_DATASET_STORAGE_KEY, $("replayDataset").value);
  fillLerobotReplayDatasets(state.overview?.lerobot_replay_datasets || []);
};
$("replayEpisode").onchange = () => {
  localStorage.setItem(REPLAY_EPISODE_STORAGE_KEY, $("replayEpisode").value);
};
$("convertButton").onclick = convertRaw;
$("safeButton").onclick = emergencySafe;
$("exitStopButton").onclick = clearEmergency;
$("restartDevicesButton").onclick = restartDevices;
$("startRobotButton").onclick = () => startRobot(false);
$("checkRobotButton").onclick = () => startRobot(true);
$("moveInitPoseButton").onclick = moveInitPose;
$("moveRlInitPoseButton").onclick = moveInitPose;
$("moveRltInitPoseButton").onclick = moveInitPose;
$("startRltMode1Button").onclick = () => startRltOnline(1);
$("startRltMode2Button").onclick = () => startRltOnline(2);
$("startRltReferenceButton").onclick = startRltReference;
$("saveRltSuccessButton").onclick = () => finalizeRlt("success");
$("saveRltFailureButton").onclick = () => finalizeRlt("failure");
$("abortRltButton").onclick = abortRlt;
$("refreshButton").onclick = refresh;
$("refreshJobsButton").onclick = refresh;
document.querySelectorAll(".module-tab").forEach((tab) => {
  tab.onclick = () => {
    activateTab(tab.dataset.tab);
    window.location.hash = tab.dataset.tab;
  };
});

const requestedTab = window.location.hash.slice(1);
const knownTabs = [...document.querySelectorAll(".module-tab")].map((tab) => tab.dataset.tab);
const savedTab = localStorage.getItem("cobot_dashboard_tab");
activateTab(knownTabs.includes(requestedTab) ? requestedTab : knownTabs.includes(savedTab) ? savedTab : "il");
renderPreviewRoute();
refresh();
startRuntimeEvents();
window.addEventListener("beforeunload", () => {
  state.eventsStopped = true;
  state.eventController?.abort();
});
// Initial snapshot and action-triggered refreshes are retained; periodic
// overview/log polling is intentionally disabled. MJPEG remains independent.
for (const name of LIVE_CAMERA_NAMES) startMjpegCamera(name); // latest-only preview, route shown in UI
