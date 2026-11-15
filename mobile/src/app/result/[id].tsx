import { router, Stack, useLocalSearchParams } from 'expo-router';
import { ScrollView, StyleSheet, Text, View } from 'react-native';

import { api, recentResults } from '../../api';
import { Button } from '../../components/Button';
import { MaskOverlay } from '../../components/MaskOverlay';
import { ErrorState, Loading } from '../../components/StateView';
import { colors, space } from '../../components/theme';
import { VerdictBadge } from '../../components/VerdictBadge';
import { percent, titleCase, when } from '../../lib/format';
import { useAsync } from '../../lib/useAsync';

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <View style={styles.stat}>
      <Text style={styles.statLabel}>{label}</Text>
      <Text style={styles.statValue}>{value}</Text>
    </View>
  );
}

export default function Result() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const numericId = Number(id);
  const { state, reload } = useAsync(async () => {
    if (!Number.isInteger(numericId)) throw new Error(`"${id}" is not an inspection id`);
    return recentResults.get(numericId) ?? api.inspection(numericId);
  }, [numericId]);

  if (state.status === 'loading') return <Loading label="Loading result…" />;
  if (state.status === 'error') return <ErrorState message={state.error} onRetry={reload} />;
  const r = state.data;

  return (
    <ScrollView contentContainerStyle={styles.page}>
      <Stack.Screen options={{ title: `Inspection #${r.id}` }} />
      <View style={styles.headline}>
        <VerdictBadge defective={r.is_defective} />
        <Text style={styles.meta}>
          {titleCase(r.category)} · {when(r.created_at)}
        </Text>
      </View>
      <MaskOverlay imageBase64={r.image_png_base64} maskLayerBase64={r.mask_layer_png_base64} />
      <View style={styles.stats}>
        <Stat label="Anomaly score" value={`${r.anomaly_score.toFixed(1)} / ${r.threshold.toFixed(1)}`} />
        <Stat label="Defect area" value={percent(r.defect_area_fraction)} />
        <Stat label="Model time" value={`${Math.round(r.latency_ms)} ms`} />
      </View>
      <Text style={styles.note}>
        The part is called defective when its score is above the cutoff learned from good parts only. Red marks the
        pixels the model thinks are defective.
      </Text>
      <Button
        label="Inspect another"
        onPress={() => router.replace({ pathname: '/inspect', params: { category: r.category } })}
      />
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  page: { padding: space(4), gap: space(4) },
  headline: { flexDirection: 'row', alignItems: 'center', gap: space(3) },
  meta: { color: colors.muted, fontSize: 14 },
  stats: { flexDirection: 'row', gap: space(2) },
  stat: {
    flex: 1,
    backgroundColor: colors.card,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: colors.border,
    padding: space(3),
    gap: 2,
  },
  statLabel: { fontSize: 12, color: colors.muted },
  statValue: { fontSize: 16, fontWeight: '600', color: colors.text, fontVariant: ['tabular-nums'] },
  note: { fontSize: 13, color: colors.muted, lineHeight: 19 },
});
