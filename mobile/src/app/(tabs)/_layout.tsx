import { Tabs } from 'expo-router/js-tabs';
import { type ColorValue, Text } from 'react-native';

import { colors } from '../../components/theme';

const icon = (glyph: string) =>
  function TabIcon({ color }: { color: ColorValue }) {
    return <Text style={{ color, fontSize: 18 }}>{glyph}</Text>;
  };

export default function TabsLayout() {
  return (
    <Tabs
      screenOptions={{
        tabBarActiveTintColor: colors.primary,
        headerTitleStyle: { color: colors.text },
        sceneStyle: { backgroundColor: colors.bg },
      }}
    >
      <Tabs.Screen name="index" options={{ title: 'MaskSight', tabBarLabel: 'Home', tabBarIcon: icon('◎') }} />
      <Tabs.Screen name="history" options={{ title: 'History', tabBarIcon: icon('☰') }} />
    </Tabs>
  );
}
