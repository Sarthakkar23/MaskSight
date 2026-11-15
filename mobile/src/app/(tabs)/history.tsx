import { router, useFocusEffect } from 'expo-router';
import { useCallback, useRef } from 'react';
import { FlatList, Pressable, RefreshControl, StyleSheet, Text, View } from 'react-native';

import { api } from '../../api';
import { Button } from '../../components/Button';
import { EmptyState, ErrorState, Loading } from '../../components/StateView';
import { colors, space } from '../../components/theme';
import { VerdictBadge } from '../../components/VerdictBadge';
import { percent, titleCase, when } from '../../lib/format';
import { useAsync } from '../../lib/useAsync';

export default function History() {
  const { state, reload } = useAsync(() => api.inspections(100));

  // Refresh whenever the tab comes back into view (e.g. after a new inspection), but not on first mount.
  const first = useRef(true);
  useFocusEffect(
    useCallback(() => {
      if (first.current) first.current = false;
      else reload();
    }, [reload]),
  );

  if (state.status === 'loading' && !state.data) return <Loading label="Loading inspections…" />;
  if (state.status === 'error' && !state.data) return <ErrorState message={state.error} onRetry={reload} />;
  const rows = state.data ?? [];

  return (
    <FlatList
      data={rows}
      keyExtractor={(r) => String(r.id)}
      contentContainerStyle={rows.length ? styles.list : styles.emptyList}
      refreshControl={<RefreshControl refreshing={state.status === 'loading'} onRefresh={reload} />}
      ListEmptyComponent={
        <EmptyState
          title="No inspections yet"
          body="Results you run are saved on the server and show up here."
          action={<Button label="Start an inspection" onPress={() => router.navigate('/')} />}
        />
      }
      renderItem={({ item }) => (
        <Pressable
          accessibilityRole="button"
          onPress={() => router.push({ pathname: '/result/[id]', params: { id: String(item.id) } })}
          style={({ pressed }) => [styles.row, pressed && { opacity: 0.7 }]}
        >
          <View style={{ flex: 1, gap: 2 }}>
            <Text style={styles.rowTitle}>
              #{item.id} · {titleCase(item.category)}
            </Text>
            <Text style={styles.rowMeta} numberOfLines={1}>
              score {item.anomaly_score.toFixed(1)} · area {percent(item.defect_area_fraction)} · {when(item.created_at)}
            </Text>
          </View>
          <VerdictBadge defective={item.is_defective} size="sm" />
        </Pressable>
      )}
    />
  );
}

const styles = StyleSheet.create({
  list: { padding: space(4), gap: space(2) },
  emptyList: { flexGrow: 1 },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: space(3),
    backgroundColor: colors.card,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: colors.border,
    padding: space(4),
  },
  rowTitle: { fontSize: 16, fontWeight: '600', color: colors.text },
  rowMeta: { fontSize: 13, color: colors.muted },
});
