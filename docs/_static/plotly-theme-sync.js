/* Keep embedded Plotly figures in sync with the sphinx-book-theme light/dark
   toggle. The pydata theme sets data-theme="light"|"dark" on <html>; this
   script watches that attribute and relayouts every .plotly-graph-div.

   Dark values mirror the html[data-theme="dark"] tokens in ross-tokens.css
   and the ross_dark Plotly template (ross/plotly_theme.py). Light mode sets
   the same keys to null so each figure falls back to the template baked into
   its committed output. */
(function () {
  "use strict";

  var DARK_LAYOUT = {
    paper_bgcolor: "#0b1826" /* --surface-page */,
    plot_bgcolor: "#0b1826",
    "font.color": "#dfe8f3" /* --text-body */,
    colorway: [
      "#4d97cb",
      "#ff9a3d",
      "#4fbf4f",
      "#ef5859",
      "#ab8ad0",
      "#a97166",
      "#ee9ad4",
      "#9a9a9a",
      "#cdcd4a",
      "#45cede",
    ] /* --plot-* dark */,
    "hoverlabel.bgcolor": "#122839" /* --surface-card */,
    "hoverlabel.bordercolor": "#24405a" /* --grid-line-strong */,
    "hoverlabel.font.color": "#dfe8f3",
    "legend.bgcolor": "rgba(18,40,57,0.80)",
    "legend.bordercolor": "#24405a",
  };

  var DARK_AXIS = {
    gridcolor: "#1b3348" /* --grid-line */,
    zerolinecolor: "#1b3348",
    linecolor: "#33587a" /* --border-strong */,
  };

  var DARK_SCENE_AXIS = {
    backgroundcolor: "#0b1826",
    gridcolor: "#24405a",
    linecolor: "#1b3348",
    zerolinecolor: "#1b3348",
  };

  var DARK_POLAR_AXIS = {
    gridcolor: "#1b3348",
    linecolor: "#1b3348",
  };

  var DARK_UPDATEMENU = {
    bgcolor: "#122839" /* --surface-card */,
    bordercolor: "#24405a" /* --grid-line-strong */,
    "font.color": "#dfe8f3" /* --text-body */,
  };

  function currentMode() {
    return document.documentElement.dataset.theme === "dark"
      ? "dark"
      : "light";
  }

  function assign(update, prefix, values, dark) {
    Object.keys(values).forEach(function (key) {
      update[prefix + key] = dark ? values[key] : null;
    });
  }

  function buildUpdate(gd, dark) {
    var update = {};
    assign(update, "", DARK_LAYOUT, dark);
    Object.keys(gd.layout || {}).forEach(function (name) {
      if (/^[xy]axis\d*$/.test(name)) {
        assign(update, name + ".", DARK_AXIS, dark);
      } else if (/^scene\d*$/.test(name)) {
        ["xaxis", "yaxis", "zaxis"].forEach(function (axis) {
          assign(update, name + "." + axis + ".", DARK_SCENE_AXIS, dark);
        });
      } else if (/^polar\d*$/.test(name)) {
        update[name + ".bgcolor"] = dark ? "#0b1826" : null;
        ["angularaxis", "radialaxis"].forEach(function (axis) {
          assign(update, name + "." + axis + ".", DARK_POLAR_AXIS, dark);
        });
      } else if (/^ternary\d*$/.test(name)) {
        update[name + ".bgcolor"] = dark ? "#0b1826" : null;
        ["aaxis", "baxis", "caxis"].forEach(function (axis) {
          assign(update, name + "." + axis + ".", DARK_POLAR_AXIS, dark);
        });
      }
    });
    ((gd.layout || {}).updatemenus || []).forEach(function (menu, i) {
      assign(update, "updatemenus[" + i + "].", DARK_UPDATEMENU, dark);
    });
    return update;
  }

  var plotlyPromise = null;

  function getPlotly() {
    if (typeof window.Plotly !== "undefined") {
      return Promise.resolve(window.Plotly);
    }
    if (!plotlyPromise && typeof window.require === "function") {
      plotlyPromise = new Promise(function (resolve) {
        window.require(
          ["plotly"],
          function (Plotly) {
            resolve(Plotly);
          },
          function () {
            plotlyPromise = null;
            resolve(null);
          }
        );
      });
    }
    return plotlyPromise || Promise.resolve(null);
  }

  function applyTheme(Plotly) {
    if (!Plotly) {
      return;
    }
    var mode = currentMode();
    document.querySelectorAll(".plotly-graph-div").forEach(function (gd) {
      var previous = gd.dataset.rossPlotTheme;
      if (previous === mode || !gd.layout) {
        return;
      }
      gd.dataset.rossPlotTheme = mode;
      if (mode === "light" && previous === undefined) {
        return;
      }
      Promise.resolve(
        Plotly.relayout(gd, buildUpdate(gd, mode === "dark"))
      ).catch(function () {
        delete gd.dataset.rossPlotTheme;
      });
    });
  }

  var scheduled = false;
  function scheduleApply() {
    if (scheduled) {
      return;
    }
    scheduled = true;
    window.requestAnimationFrame(function () {
      scheduled = false;
      getPlotly().then(applyTheme);
    });
  }

  new MutationObserver(scheduleApply).observe(document.documentElement, {
    attributes: true,
    attributeFilter: ["data-theme"],
  });

  function watchLateRenderedPlots() {
    new MutationObserver(scheduleApply).observe(document.body, {
      childList: true,
      subtree: true,
    });
    scheduleApply();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", watchLateRenderedPlots);
  } else {
    watchLateRenderedPlots();
  }
})();
