import { fireEvent, render, screen } from '@testing-library/react-native';

import { MaskOverlay } from '../src/components/MaskOverlay';
import { ErrorState } from '../src/components/StateView';
import { VerdictBadge } from '../src/components/VerdictBadge';

describe('components', () => {
  it('MaskOverlay layers the mask over the image at the chosen opacity', async () => {
    await render(<MaskOverlay imageBase64="IMG" maskLayerBase64="MASK" initialOpacity={0.4} />);
    const layer = screen.getByTestId('mask-layer');
    expect(layer.props.source).toEqual({ uri: 'data:image/png;base64,MASK' });
    expect(layer.props.style).toEqual(expect.arrayContaining([{ opacity: 0.4 }]));
    expect(screen.getByText('40%')).toBeTruthy();

    await fireEvent(screen.getByTestId('opacity-slider'), 'valueChange', 0.9);
    expect(screen.getByTestId('mask-layer').props.style).toEqual(expect.arrayContaining([{ opacity: 0.9 }]));
    expect(screen.getByText('90%')).toBeTruthy();
  });

  it('ErrorState shows the message and retries', async () => {
    const onRetry = jest.fn();
    await render(<ErrorState message="Server unreachable at http://h" onRetry={onRetry} />);
    expect(screen.getByText('Server unreachable at http://h')).toBeTruthy();
    await fireEvent.press(screen.getByText('Try again'));
    expect(onRetry).toHaveBeenCalledTimes(1);
  });

  it('VerdictBadge labels the verdict', async () => {
    await render(<VerdictBadge defective={false} />);
    expect(screen.getByText('Good')).toBeTruthy();
  });
});
