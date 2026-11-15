import Slider from '@react-native-community/slider';
import { useState } from 'react';
import { Image, StyleSheet, Text, View } from 'react-native';

import { colors, space } from './theme';

interface Props {
  imageBase64: string;
  maskLayerBase64: string;
  initialOpacity?: number;
}

const png = (b64: string) => ({ uri: `data:image/png;base64,${b64}` });

/** The inspected image with the predicted defect mask on top, plus a slider for the mask's opacity. */
export function MaskOverlay({ imageBase64, maskLayerBase64, initialOpacity = 0.55 }: Props) {
  const [opacity, setOpacity] = useState(initialOpacity);
  return (
    <View style={styles.wrap}>
      <View style={styles.frame}>
        <Image source={png(imageBase64)} style={StyleSheet.absoluteFill} accessibilityLabel="Inspected image" />
        <Image
          testID="mask-layer"
          source={png(maskLayerBase64)}
          style={[StyleSheet.absoluteFill, { opacity }]}
          accessibilityLabel="Predicted defect mask"
        />
      </View>
      <View style={styles.row}>
        <Text style={styles.label}>Mask</Text>
        <Slider
          testID="opacity-slider"
          style={styles.slider}
          minimumValue={0}
          maximumValue={1}
          step={0.05}
          value={initialOpacity}
          onValueChange={setOpacity}
          minimumTrackTintColor={colors.defect}
          maximumTrackTintColor={colors.border}
          thumbTintColor={colors.defect}
          accessibilityLabel="Mask opacity"
        />
        <Text style={styles.value}>{Math.round(opacity * 100)}%</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: { gap: space(2) },
  frame: {
    width: '100%',
    aspectRatio: 1,
    borderRadius: 12,
    overflow: 'hidden',
    backgroundColor: '#000',
  },
  row: { flexDirection: 'row', alignItems: 'center', gap: space(2) },
  label: { fontSize: 14, color: colors.muted, width: 40 },
  slider: { flex: 1, height: 40 },
  value: { fontSize: 14, color: colors.text, width: 44, textAlign: 'right', fontVariant: ['tabular-nums'] },
});
