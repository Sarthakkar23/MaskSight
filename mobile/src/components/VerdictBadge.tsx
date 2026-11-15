import { StyleSheet, Text, View } from 'react-native';

import { colors, space } from './theme';

export function VerdictBadge({ defective, size = 'md' }: { defective: boolean; size?: 'sm' | 'md' }) {
  return (
    <View
      style={[styles.badge, { backgroundColor: defective ? colors.defectBg : colors.okBg }, size === 'sm' && styles.sm]}
    >
      <Text style={[styles.text, { color: defective ? colors.defect : colors.ok }, size === 'sm' && styles.smText]}>
        {defective ? 'Defective' : 'Good'}
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  badge: { alignSelf: 'flex-start', borderRadius: 999, paddingHorizontal: space(3), paddingVertical: space(1) },
  sm: { paddingHorizontal: space(2), paddingVertical: 2 },
  text: { fontSize: 15, fontWeight: '700' },
  smText: { fontSize: 12 },
});
