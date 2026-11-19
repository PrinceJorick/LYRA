import type { Caregiver, NotificationResult } from '../../types';

/** Replace with a real SMS/push provider later. Nothing is ever sent today. */
export interface NotificationService {
  notifyCaregivers(caregivers: Caregiver[]): Promise<NotificationResult>;
}

export const mockNotificationService: NotificationService = {
  async notifyCaregivers(caregivers) {
    return {
      simulated: true,
      at: Date.now(),
      // Sending and delivery are simulated; acknowledgment is deliberately NOT, because the
      // proposal requires distinguishing sent / delivered / acknowledged.
      recipients: caregivers.map((c) => ({
        caregiverName: c.name,
        sending: 'simulated',
        delivery: 'simulated',
        acknowledgment: 'not_simulated',
      })),
    };
  },
};

export const notificationService: NotificationService = mockNotificationService;
