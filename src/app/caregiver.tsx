import { router, useLocalSearchParams } from 'expo-router';
import React, { useState } from 'react';
import { Button, Field, Screen, Txt } from '../components/ui';
import { useApp } from '../state/AppState';
import { newId } from '../utils/format';

export default function CaregiverForm() {
  const { id } = useLocalSearchParams<{ id?: string }>();
  const { caregivers, upsertCaregiver, removeCaregiver } = useApp();
  const existing = caregivers.find((c) => c.id === id);

  const [name, setName] = useState(existing?.name ?? '');
  const [relationship, setRelationship] = useState(existing?.relationship ?? '');
  const [phone, setPhone] = useState(existing?.phone ?? '');
  const [errors, setErrors] = useState<{ name?: string; phone?: string }>({});
  const [confirmRemove, setConfirmRemove] = useState(false);

  const save = () => {
    const e: typeof errors = {};
    if (!name.trim()) e.name = 'Enter a name.';
    if (phone.replace(/\D/g, '').length < 7) e.phone = 'Enter a phone number with at least 7 digits.';
    setErrors(e);
    if (e.name || e.phone) return;
    upsertCaregiver({
      id: existing?.id ?? newId('cg'),
      name: name.trim(), relationship: relationship.trim(), phone: phone.trim(),
    });
    router.back();
  };

  return (
    <Screen
      keyboard
      footer={
        <>
          <Button label={existing ? 'Save Changes' : 'Add Caregiver'} onPress={save} />
          {existing ? (
            <Button
              kind="danger"
              label={confirmRemove ? 'Tap again to confirm removal' : 'Remove contact'}
              onPress={() => {
                if (!confirmRemove) return setConfirmRemove(true);
                removeCaregiver(existing.id);
                router.back();
              }}
            />
          ) : null}
        </>
      }
    >
      <Txt muted>This contact is stored on this device for the demo only. No messages or calls are made.</Txt>
      <Field label="Name" value={name} onChangeText={setName} error={errors.name} autoCapitalize="words" textContentType="name" />
      <Field label="Relationship (optional)" value={relationship} onChangeText={setRelationship} autoCapitalize="words" />
      <Field label="Phone number" value={phone} onChangeText={setPhone} error={errors.phone} keyboardType="phone-pad" textContentType="telephoneNumber" />
    </Screen>
  );
}
