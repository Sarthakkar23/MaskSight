/** Shapes returned by the MaskSight API (serve/api.py). */

export interface Category {
  name: string;
  model: string;
  image_threshold: number;
}

export interface InspectionSummary {
  id: number;
  created_at: string;
  category: string;
  anomaly_score: number;
  threshold: number;
  is_defective: boolean;
  defect_area_fraction: number;
  latency_ms: number;
  filename: string | null;
}

export interface InspectionDetail extends InspectionSummary {
  /** 224 x 224 centre crop the model saw, PNG. */
  image_png_base64: string;
  /** Binary mask, white = defect, PNG. */
  mask_png_base64: string;
  /** Transparent PNG, coloured where the mask is set; laid over the image. */
  mask_layer_png_base64: string;
}

/** A picked photo, in the form the upload needs on each platform. */
export interface PickedImage {
  uri: string;
  name: string;
  mimeType: string;
  /** Present on web, where the picker hands back a File. */
  file?: Blob;
}
