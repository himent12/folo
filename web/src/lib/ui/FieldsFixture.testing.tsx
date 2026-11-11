/** Every shared control inside a Field, with a hint or an error, for `fields.test.tsx`. */
import { signal } from 'fw';
import Field from './Field.tsx';
import Input from './Input';
import Segmented from './Segmented';
import Select from './Select';
import Slider from './Slider';
import Switch from './Switch';

export default function FieldsFixture(props: { error?: string | null }) {
  const error = () => props.error ?? null;
  const seg = signal('a');
  const sel = signal('a');
  const on = signal(false);
  const level = signal(0.5);
  const options = [
    { value: 'a', label: 'A' },
    { value: 'b', label: 'B' },
  ];
  return (
    <>
      <Field label="Name" for="f-input" hint="Letters and digits." error={error()}>
        <Input id="f-input" aria-describedby="extra" />
      </Field>
      <Field label="Size" for="f-seg" hint="Per batch." error={error()}>
        <Segmented id="f-seg" label="Size" bind:value={seg} options={options} />
      </Field>
      <Field label="Model" for="f-select" hint="Trained on shapes." error={error()}>
        <Select id="f-select" bind:value={sel} options={options} />
      </Field>
      <Field label="Mode" hint="Without an id of its own." error={error()}>
        <Segmented label="Mode" bind:value={seg} options={options} />
      </Field>
      <Field label="Fast" for="f-switch" hint="Uses tensor cores." error={error()}>
        <Switch id="f-switch" label="Fast" hint="On the GPU." bind:checked={on} />
      </Field>
      <Field label="Level" hint="How much." error={error()}>
        <Slider label="Level" bind:value={level} />
      </Field>
    </>
  );
}
