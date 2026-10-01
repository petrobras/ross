// Stretching the rotor figure vertically, and saying so on the axis.
//
// WHY THIS EXISTS. `plot_rotor` locks the drawing at 1:1 (`scaleanchor: 'x'`,
// `scaleratio: 1`) on purpose, and for a slender rotor that is a tyranny of
// arithmetic rather than a choice anyone made: a 1.5 m rotor with a 22 cm
// vertical envelope is 7.5:1, so at the ~645 px of plot width a laptop gives,
// the drawing is **86 px tall**. Nothing about the height setting can change
// that -- with `constrain: 'domain'` the size of the drawing is decided by the
// width. The only ways out are more width or a broken ratio.
//
// So the ratio is broken, deliberately, by the amount the person asks for --
// and the axis says so. `scaleratio: 3` means one unit of y is worth three
// units of x in pixels, which is exactly "three times vertical exaggeration",
// the same convention a geological section uses.
//
// WHY THE LABEL IS NOT OPTIONAL. An engineer reading a rotor and not suspecting
// the scale draws the wrong conclusion about slenderness, which is the one
// property the picture is there to convey. A stretched figure with no label is
// the only dishonest thing this module could do, so the label is written by the
// same function that does the stretching and there is no way to have one
// without the other.

// What the control offers. `1` first and default, so a person who never touches
// it sees the true proportions.
export const VERTICAL_SCALES = [1, 2, 5];

// Where something placed outside the plot area has to go for it to stay the
// same number of pixels away after the area grows `times` times.
//
// ROSS places the toggle buttons and the legend in **paper** coordinates, which
// are fractions of the plot area's height: the buttons at `-0.4615` of a 130 px
// area are 60 px under the axis, inside the band its bottom margin reserves for
// them, and the legend at `1.4615` is 60 px above it.
//
// Leave those fractions alone while the area grows and the offsets grow with
// it -- past the margins ROSS reserved. What happens then was first *guessed*
// here ("the buttons fall off the figure") and then measured in a browser, and
// the guess was wrong: Plotly's `margin.autoexpand` keeps them on the figure by
// **shrinking the plot area** to make room. At 5× on a 1.5 m rotor that took
// the x axis from 646 px to 614 -- the rotor came out narrower, the one thing
// a vertical stretch must never do -- and pushed the legend up over the title.
//
// So the fraction is divided by the factor below the area and above it, and
// nothing inside [0, 1] is touched: those are positions *on* the drawing, which
// should move with it.
function keepPixelOffset(y, times) {
    if (typeof y !== 'number') return y;
    if (y < 0) return y / times;
    if (y > 1) return 1 + (y - 1) / times;
    return y;
}

function keepOffsetsOutsideTheArea(layout, times) {
    (layout.updatemenus || []).forEach(menu => {
        menu.y = keepPixelOffset(menu.y, times);
    });
    if (layout.legend) layout.legend.y = keepPixelOffset(layout.legend.y, times);
    // Only what is anchored to the paper. An annotation in data coordinates --
    // the node scale label, at a negative *radius* -- is part of the drawing and
    // stretches with it. The axes indicator is anchored at the edge (y = 0) with
    // pixel shifts, which this leaves exactly where it was.
    (layout.annotations || []).forEach(note => {
        if (note.yref === 'paper') note.y = keepPixelOffset(note.y, times);
    });
}

function stretchTitle(axis, note) {
    const current = axis.title;
    const base = (current && typeof current === 'object' ? current.text : current) || '';
    axis.title = { text: base ? base + ' — ' + note : note };
}

// ROSS's layout with the drawing stretched `factor` times vertically.
//
// `note` arrives already translated: this module has no business knowing which
// language the screen is in, and the phrase belongs with the other phrases.
//
// **What goes in is ROSS's layout**, and what comes out is a new one: the
// argument is never touched, so the figure the server sent stays pristine and
// can be stretched again at another factor without carrying the first one.
//
// That is the property, and it is deliberately not idempotence. Stretching an
// already stretched layout is meaningless -- the note would stack and so would
// the height -- and the honest answer is to make the caller unable to do it by
// always handing back a copy, rather than to half-detect it. A first version
// stripped a previous note and left the height compounding: correct-looking and
// wrong, which is worse than plainly wrong.
export function withVerticalScale(layout, factor, note) {
    const stretched = JSON.parse(JSON.stringify(layout || {}));
    const times = Number(factor);
    if (!isFinite(times) || times <= 1) return stretched;

    const axis = stretched.yaxis = stretched.yaxis || {};
    // The knob Plotly already has. Replacing the ranges by hand would be a
    // second implementation of what `scaleanchor` does, and a worse one: it
    // would drift the moment the container changed width.
    axis.scaleratio = times;

    // The plot area has to grow with the factor, or Plotly honours the ratio by
    // shrinking the **x** domain instead -- the rotor would come out narrower
    // rather than taller, which is the opposite of what was asked.
    //
    // ROSS's height is `plot area + top + bottom`, and its margins are left
    // alone: they reserve fixed bands for the axis title, the toggle buttons
    // and the axes indicator.
    const margin = stretched.margin || {};
    const top = margin.t || 0;
    const bottom = margin.b || 0;
    const area = (stretched.height || 0) - top - bottom;
    if (area > 0) stretched.height = area * times + top + bottom;

    keepOffsetsOutsideTheArea(stretched, times);
    stretchTitle(axis, note);
    return stretched;
}
