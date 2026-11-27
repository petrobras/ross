// What is being computed right now, in one place.
//
// WHY THIS EXISTS. A card that knows only about itself cannot tell waiting
// from computing. Asked to run the Campbell and the crack together, both cards
// would show "updating", and read as two analyses running in parallel. There is
// one worker and one thread -- they take turns -- and nothing on screen could
// say so without knowing about more than one card.
//
// So the line has a single reader here. Each job announces itself while it runs,
// the bar at the bottom of the page subscribes, and the card gets the state that
// belongs to it. `core/` and not `features/`: it holds no DOM and knows nothing
// about how any of it is drawn.
//
// WHY A RUN TOKEN. A card that is run again aborts its own previous request,
// and the old `runJob` then unwinds -- after the new one has already announced
// itself. Without an identity to compare, the loser's cleanup would erase the
// winner's entry and the bar would go quiet with an analysis still running.

const IN_FLIGHT = new Map();
const WATCHERS = [];

export function watchWork(listen) {
    WATCHERS.push(listen);
    listen(workNow());
}

export function workNow() {
    return Array.from(IN_FLIGHT.values());
}

export function noteWork(subject, run, progress) {
    IN_FLIGHT.set(subject, Object.assign({ subject, run }, progress));
    announce();
}

// Only the run that owns the entry may remove it. See "why a run token" above.
export function forgetWork(subject, run) {
    const current = IN_FLIGHT.get(subject);
    if (!current || current.run !== run) return;
    IN_FLIGHT.delete(subject);
    announce();
}

function announce() {
    const now = workNow();
    WATCHERS.forEach(listen => listen(now));
}
