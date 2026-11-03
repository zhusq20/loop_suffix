import { exportToSvg, exportToBlob } from "@excalidraw/excalidraw";
window.runExport = async (scene, scale, padding) => {
  const opts = {
    elements: scene.elements,
    appState: { ...(scene.appState || {}), exportBackground: true,
                viewBackgroundColor: "#ffffff", exportWithDarkMode: false },
    files: scene.files || {},
    exportPadding: padding,
  };
  const svg = await exportToSvg(opts);
  const blob = await exportToBlob({ ...opts, mimeType: "image/png",
    getDimensions: (w, h) => ({ width: w * scale, height: h * scale, scale }) });
  const bytes = new Uint8Array(await blob.arrayBuffer());
  let bin = ""; for (let i = 0; i < bytes.length; i += 0x8000)
    bin += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  return { svg: svg.outerHTML, png: btoa(bin) };
};
