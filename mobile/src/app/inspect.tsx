import * as ImagePicker from 'expo-image-picker';
import { router, Stack, useLocalSearchParams } from 'expo-router';
import { useState } from 'react';
import { Image, Platform, ScrollView, StyleSheet, Text, View } from 'react-native';

import { api, recentResults } from '../api';
import type { PickedImage } from '../api';
import { Button } from '../components/Button';
import { EmptyState } from '../components/StateView';
import { colors, space } from '../components/theme';
import { titleCase } from '../lib/format';

function toPicked(asset: ImagePicker.ImagePickerAsset): PickedImage {
  const mimeType = asset.mimeType ?? 'image/jpeg';
  const ext = mimeType.split('/')[1] ?? 'jpg';
  return { uri: asset.uri, name: asset.fileName ?? `photo.${ext}`, mimeType, file: asset.file };
}

export default function Inspect() {
  const { category } = useLocalSearchParams<{ category?: string }>();
  const [image, setImage] = useState<PickedImage | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!category) {
    return (
      <EmptyState
        title="No category chosen"
        body="Pick a product on the home screen first."
        action={<Button label="Go home" onPress={() => router.navigate('/')} />}
      />
    );
  }

  async function pick(source: 'camera' | 'library') {
    setError(null);
    if (source === 'camera' && Platform.OS !== 'web') {
      const perm = await ImagePicker.requestCameraPermissionsAsync();
      if (!perm.granted) {
        setError('Camera access was denied. Allow it in Settings, or choose a photo from the gallery.');
        return;
      }
    }
    const options: ImagePicker.ImagePickerOptions = { mediaTypes: ['images'], quality: 0.9 };
    const result =
      source === 'camera'
        ? await ImagePicker.launchCameraAsync(options)
        : await ImagePicker.launchImageLibraryAsync(options);
    if (!result.canceled && result.assets[0]) setImage(toPicked(result.assets[0]));
  }

  async function upload() {
    if (!image || !category) return;
    setBusy(true);
    setError(null);
    try {
      const res = await api.predict(category, image);
      recentResults.set(res.id, res);
      router.replace({ pathname: '/result/[id]', params: { id: String(res.id) } });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
    } finally {
      setBusy(false);
    }
  }

  return (
    <ScrollView contentContainerStyle={styles.page}>
      <Stack.Screen options={{ title: `Inspect ${titleCase(category)}` }} />
      <View style={styles.preview}>
        {image ? (
          <Image source={{ uri: image.uri }} style={StyleSheet.absoluteFill} resizeMode="cover" />
        ) : (
          <Text style={styles.placeholder}>
            Photograph one {titleCase(category).toLowerCase()} on a plain background, filling most of the frame.
          </Text>
        )}
      </View>
      <View style={styles.row}>
        <View style={{ flex: 1 }}>
          <Button label="Take photo" variant="secondary" onPress={() => pick('camera')} disabled={busy} />
        </View>
        <View style={{ flex: 1 }}>
          <Button label="From gallery" variant="secondary" onPress={() => pick('library')} disabled={busy} />
        </View>
      </View>
      {error && (
        <Text style={styles.error} accessibilityRole="alert">
          {error}
        </Text>
      )}
      <Button label="Inspect" onPress={upload} disabled={!image} busy={busy} />
      {busy && <Text style={styles.hint}>Uploading and running the model…</Text>}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  page: { padding: space(4), gap: space(4) },
  preview: {
    width: '100%',
    aspectRatio: 1,
    borderRadius: 12,
    borderWidth: 1,
    borderStyle: 'dashed',
    borderColor: colors.border,
    backgroundColor: colors.card,
    overflow: 'hidden',
    alignItems: 'center',
    justifyContent: 'center',
    padding: space(6),
  },
  placeholder: { color: colors.muted, fontSize: 15, textAlign: 'center' },
  row: { flexDirection: 'row', gap: space(3) },
  error: { color: colors.defect, fontSize: 14 },
  hint: { color: colors.muted, fontSize: 14, textAlign: 'center' },
});
