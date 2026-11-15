import { router } from 'expo-router';
import { FlatList, Pressable, RefreshControl, StyleSheet, Text, View } from 'react-native';

import { api } from '../../api';
import { EmptyState, ErrorState, Loading } from '../../components/StateView';
import { colors, space } from '../../components/theme';
import { titleCase } from '../../lib/format';
import { useAsync } from '../../lib/useAsync';

export default function Home() {
  const { state, reload } = useAsync(api.categories);

  if (state.status === 'loading' && !state.data) return <Loading label="Connecting to the MaskSight server…" />;
  if (state.status === 'error' && !state.data) return <ErrorState message={state.error} onRetry={reload} />;
  const categories = state.data ?? [];

  return (
    <FlatList
      data={categories}
      keyExtractor={(c) => c.name}
      contentContainerStyle={styles.list}
      refreshControl={<RefreshControl refreshing={state.status === 'loading'} onRefresh={reload} />}
      ListHeaderComponent={
        <View style={styles.header}>
          <Text style={styles.title}>What are you inspecting?</Text>
          <Text style={styles.subtitle}>
            Each product has its own model, trained only on photos of good parts. Pick one, then photograph a part.
          </Text>
        </View>
      }
      ListEmptyComponent={
        <EmptyState title="No models loaded" body={`The server at ${api.baseUrl} has no category checkpoints.`} />
      }
      renderItem={({ item }) => (
        <Pressable
          accessibilityRole="button"
          onPress={() => router.push({ pathname: '/inspect', params: { category: item.name } })}
          style={({ pressed }) => [styles.card, pressed && styles.pressed]}
        >
          <View style={{ flex: 1 }}>
            <Text style={styles.cardTitle}>{titleCase(item.name)}</Text>
            <Text style={styles.cardMeta}>
              {item.model} · cutoff {item.image_threshold.toFixed(1)}
            </Text>
          </View>
          <Text style={styles.chevron}>›</Text>
        </Pressable>
      )}
    />
  );
}

const styles = StyleSheet.create({
  list: { padding: space(4), gap: space(3) },
  header: { gap: space(1), marginBottom: space(2) },
  title: { fontSize: 22, fontWeight: '700', color: colors.text },
  subtitle: { fontSize: 15, color: colors.muted, lineHeight: 21 },
  card: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.card,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: colors.border,
    padding: space(4),
  },
  pressed: { opacity: 0.7 },
  cardTitle: { fontSize: 17, fontWeight: '600', color: colors.text },
  cardMeta: { fontSize: 13, color: colors.muted, marginTop: 2 },
  chevron: { fontSize: 26, color: colors.muted },
});
