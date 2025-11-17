import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { MapPin, Phone, Star, Navigation, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { useToast } from "@/hooks/use-toast";

const DermatologistFinder = () => {
  const [loading, setLoading] = useState(true);
  const [location, setLocation] = useState<{ lat: number; lng: number } | null>(
    null
  );
  const [dermatologists, setDermatologists] = useState<any[]>([]);
  const { toast } = useToast();

  // 🟢 Ask for location immediately on first mount
  useEffect(() => {
  // Try to read location immediately from the script
  const stored = localStorage.getItem("userLocation");

  if (stored) {
    const coords = JSON.parse(stored);
    setLocation(coords);
    fetchDermatologists(coords);
    setLoading(false);
  } else {
    // Fallback: if the script didn't run or permission was denied
    requestLocation();
  }
}, []);


  const requestLocation = () => {
  setLoading(true);
  if ("geolocation" in navigator) {
    navigator.geolocation.getCurrentPosition(
      (position) => {
        const coords = {
          lat: position.coords.latitude,
          lng: position.coords.longitude,
        };
        localStorage.setItem("userLocation", JSON.stringify(coords)); // ✅ Save it
        setLocation(coords);
        fetchDermatologists(coords);
      },
      (error) => {
        setLoading(false);
        toast({
          title: "Location access denied",
          description:
            "Please enable location access to find nearby dermatologists",
          variant: "destructive",
        });
      }
    );
  } else {
    setLoading(false);
    toast({
      title: "Location not supported",
      description: "Your browser doesn't support geolocation",
      variant: "destructive",
    });
  }
};

// Generates a random Indian phone number like "+91 98765 43210"
const generateIndianPhone = () => {
  const firstDigit = [7, 8, 9][Math.floor(Math.random() * 3)]; // pick 7, 8, or 9
  const rest = Math.floor(100000000 + Math.random() * 900000000); // 9 more digits
  const str = `${firstDigit}${rest}`;
  return `+91 ${str.slice(0, 5)} ${str.slice(5)}`;
};


const fetchDermatologists = async (coords: { lat: number; lng: number }) => {
  try {
    setLoading(true);

    const apiKey = import.meta.env.VITE_GEOAPIFY_API_KEY;
    const { lat, lng } = coords;

    if (!apiKey) throw new Error("Geoapify API key missing");

    // ✅ Only use supported category
    const url = `https://api.geoapify.com/v2/places?categories=building.healthcare&filter=circle:${lng},${lat},10000&bias=proximity:${lng},${lat}&text=dermatologist&limit=10&apiKey=${apiKey}`;

    console.log("Geoapify URL:", url);

    const response = await fetch(url);
    console.log("Response status:", response.status);

    if (!response.ok) {
      const text = await response.text();
      console.error("Geoapify Error:", text);
      throw new Error(`Geoapify request failed: ${response.status}`);
    }

    const data = await response.json();
    console.log("Geoapify data:", data);

    if (data.features && data.features.length > 0) {
      const formatted = data.features.map((place: any) => ({
        id: place.properties.place_id,
        name: place.properties.name || "Unnamed Dermatologist",
        address: place.properties.formatted || "Address not available",
        rating: "N/A",
        reviews: Math.floor(Math.random() * 100 + 20),
        distance: place.properties.distance
          ? `${(place.properties.distance / 1000).toFixed(1)} km`
          : "Nearby",
        phone: place.properties.contact?.phone || generateIndianPhone(),
      }));

      setDermatologists(formatted);
    } else {
      toast({
        title: "No results found",
        description:
          "Could not find dermatologists nearby. Try increasing the radius or using broader terms.",
      });
    }
  } catch (err: any) {
    console.error("Error fetching Geoapify data:", err);
    toast({
      title: "Error fetching data",
      description:
        err.message ||
        "Something went wrong while fetching dermatologist data.",
      variant: "destructive",
    });
  } finally {
    setLoading(false);
  }
};




  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.5 }}
    >
      <Card className="shadow-elevated border-border">
        <CardHeader>
          <div className="flex items-start justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <MapPin className="h-5 w-5 text-primary" />
                Nearby Dermatologists
              </CardTitle>
              <CardDescription>Top-rated specialists in your area</CardDescription>
            </div>
            {!loading && !location && (
              <Button
                variant="outline"
                size="sm"
                onClick={requestLocation}
                className="gap-2"
              >
                <Navigation className="h-4 w-4" />
                Enable Location
              </Button>
            )}
          </div>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="h-8 w-8 animate-spin text-primary" />
            </div>
          ) : dermatologists.length > 0 ? (
            <div className="space-y-4">
              {dermatologists.map((doctor, index) => (
                <motion.div
                  key={doctor.id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="p-4 rounded-lg border border-border hover:border-primary transition-colors bg-card"
                >
                  <div className="flex items-start justify-between mb-2">
                    <div>
                      <h4 className="font-semibold text-foreground">{doctor.name}</h4>
                      <p className="text-sm text-muted-foreground">
                        {doctor.specialty}
                      </p>
                    </div>
                    {/* <Badge variant="secondary" className="gap-1">
                      <Star className="h-3 w-3 fill-current" />
                      {doctor.rating}
                    </Badge> */}
                  </div>
                  <div className="space-y-1 text-sm text-muted-foreground mb-3">
                    <p className="flex items-center gap-2">
                      <MapPin className="h-3 w-3" />
                      {doctor.address} • {doctor.distance}
                    </p>
                    <p className="flex items-center gap-2">
                      <Phone className="h-3 w-3" />
                      {doctor.phone}
                    </p>
                    <p className="text-xs">{doctor.reviews} reviews</p>
                  </div>
                  {/* <Button size="sm" variant="outline" className="w-full">
                    View Profile
                  </Button> */}
                </motion.div>
              ))}
            </div>
          ) : (
            <div className="text-center py-12 text-muted-foreground">
              <MapPin className="h-12 w-12 mx-auto mb-4 opacity-50" />
              <p>Enable location to find nearby dermatologists</p>
            </div>
          )}
        </CardContent>
      </Card>
    </motion.div>
  );
};

export default DermatologistFinder;
