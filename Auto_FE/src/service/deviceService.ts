import { Platform } from 'react-native';
import DeviceInfo from 'react-native-device-info';
import { API_BASE } from '../api';

export const registerDeviceToken = async (accessToken: string, fcmToken: string) => {
  const deviceId = DeviceInfo.getDeviceId();
  const deviceName = await DeviceInfo.getDeviceName();
  const deviceType = Platform.OS === 'ios' ? 'IOS' : 'ANDROID';
  console.log('Registering device token:', {
    fcmToken,
    deviceId,
    deviceType,
    deviceName
  });
    
  const response = await fetch(`${API_BASE}/device-token/register`, {
    
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${accessToken}`
    },
    body: JSON.stringify({
      fcmToken: fcmToken,
      deviceId: deviceId,
      deviceType: deviceType,
      deviceName: deviceName
    })
  });

  if (!response.ok) {
    throw new Error('Failed to register device token');
  }

  return await response.json();
};
