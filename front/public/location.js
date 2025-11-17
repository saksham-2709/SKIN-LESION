// public/location-init.js
if ('geolocation' in navigator) {
  navigator.geolocation.getCurrentPosition(
    (position) => {
      localStorage.setItem(
        'userLocation',
        JSON.stringify({
          lat: position.coords.latitude,
          lng: position.coords.longitude,
        })
      );
    },
    (error) => {
      console.warn('Location permission denied or unavailable:', error);
    }
  );
}
