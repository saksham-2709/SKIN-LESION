import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Upload, User, FileText, ArrowRight, ArrowLeft, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Checkbox } from "@/components/ui/checkbox";
import { useToast } from "@/hooks/use-toast";
import ImageUpload from "./ImageUpload";

const SIDE_EFFECTS = [
  "Itching",
  "Burning sensation",
  "Pain",
  "Bleeding",
  "Swelling",
  "Redness",
  "Dry skin",
  "Pus or discharge",
  "Scaling",
  "Crusting",
];

interface DiagnosticFormProps {
  onComplete: (data: any) => void;
}

const DiagnosticForm = ({ onComplete }: DiagnosticFormProps) => {
  const [currentStep, setCurrentStep] = useState(1);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const { toast } = useToast();

  const [formData, setFormData] = useState({
    name: "",
    age: "",
    gender: "",
    areaInfected: "",
    sideEffects: [] as string[],
    image: null as File | null,
    imagePreview: "",
  });

  const handleImageUpload = (file: File, preview: string) => {
    setFormData({ ...formData, image: file, imagePreview: preview });
  };

  const handleSideEffectToggle = (effect: string) => {
    const current = formData.sideEffects;
    if (current.includes(effect)) {
      setFormData({ ...formData, sideEffects: current.filter(e => e !== effect) });
    } else {
      setFormData({ ...formData, sideEffects: [...current, effect] });
    }
  };

  // ✅ Handle "Others" checkbox
  const handleOthersToggle = (checked: boolean) => {
    setShowOtherInput(checked);
    if (!checked) {
      // Remove the previous "otherSymptom" value if unchecked
      setOtherSymptom("");
      setFormData({
        ...formData,
        sideEffects: formData.sideEffects.filter(
          (effect) => effect !== otherSymptom && effect !== "Others"
        ),
      });
    } else {
      // Add "Others" placeholder for now
      if (!formData.sideEffects.includes("Others")) {
        setFormData({
          ...formData,
          sideEffects: [...formData.sideEffects, "Others"],
        });
      }
    }
  };

  const [showOtherInput, setShowOtherInput] = useState(false);
  const [otherSymptom, setOtherSymptom] = useState("");

  // ✅ Handle text input for "Others" symptom
  const handleOtherInputChange = (value: string) => {
    setOtherSymptom(value);
    const current = formData.sideEffects.filter(
      (effect) => effect !== otherSymptom && effect !== "Others"
    );

    if (value.trim().length > 0) {
      setFormData({
        ...formData,
        sideEffects: [...current, value.trim()],
      });
    } else {
      setFormData({ ...formData, sideEffects: current });
    }
  };

  const handleSubmit = async () => {
    if (!formData.image) {
      toast({
        title: "Image required",
        description: "Please upload an image for analysis",
        variant: "destructive",
      });
      return;
    }

    setIsAnalyzing(true);

    try {
      // Create FormData to send to Flask backend
      const formDataToSend = new FormData();
      formDataToSend.append('image', formData.image);
      formDataToSend.append('name', formData.name);
      formDataToSend.append('age', formData.age);
      formDataToSend.append('gender', formData.gender);
      formDataToSend.append('areaInfected', formData.areaInfected);
      formData.sideEffects.forEach((effect) => {
        formDataToSend.append('sideEffects', effect);
      });

      // API endpoint - default to localhost:5000, can be configured via env
      const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';
      
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formDataToSend,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
        throw new Error(errorData.error || `Server error: ${response.status}`);
      }

      const result = await response.json();
      
      // Combine API response with form data for ResultsSection
      const completeResults = {
        ...formData,
        diagnosis: result.diagnosis,
        confidence: result.confidence,
        gradCamUrl: result.gradCamUrl,
        causes: result.causes || [],
        remedies: result.remedies || [],
        relatedSideEffects: result.relatedSideEffects || [],
        classCode: result.classCode,
        allPredictions: result.allPredictions || {},
        userData: result.userData || {},
      };

      setIsAnalyzing(false);
      onComplete(completeResults);
    } catch (error: any) {
      console.error('Prediction error:', error);
      setIsAnalyzing(false);
      toast({
        title: "Analysis failed",
        description: error.message || "Failed to analyze image. Please make sure the backend server is running.",
        variant: "destructive",
      });
    }
  };

  const nextStep = () => {
    if (currentStep === 1 && (!formData.name || !formData.age || !formData.gender || !formData.areaInfected)) {
      toast({
        title: "Missing information",
        description: "Please fill in all personal details",
        variant: "destructive",
      });
      return;
    }
    if (currentStep === 2 && !formData.image) {
      toast({
        title: "Image required",
        description: "Please upload an image for analysis",
        variant: "destructive",
      });
      return;
    }
    setCurrentStep(prev => Math.min(prev + 1, 3));
  };

  const prevStep = () => setCurrentStep(prev => Math.max(prev - 1, 1));

  return (
    <div className="max-w-4xl mx-auto">
      {/* Progress Indicator */}
      <div className="mb-8">
        <div className="flex justify-between items-center mb-2">
          {[1, 2, 3].map((step) => (
            <div key={step} className="flex items-center flex-1">
              <div className={`
                flex items-center justify-center w-10 h-10 rounded-full font-semibold
                ${currentStep >= step ? 'bg-primary text-primary-foreground' : 'bg-muted text-muted-foreground'}
              `}>
                {step}
              </div>
              {step < 3 && (
                <div className={`flex-1 h-1 mx-2 rounded ${currentStep > step ? 'bg-primary' : 'bg-muted'}`} />
              )}
            </div>
          ))}
        </div>
        <div className="flex justify-between text-xs text-muted-foreground mt-2">
          <span>Personal Details</span>
          <span>Image Upload</span>
          <span>Symptoms</span>
        </div>
      </div>

      <AnimatePresence mode="wait">
        {currentStep === 1 && (
          <motion.div
            key="step1"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
          >
            <Card className="shadow-elevated border-border">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <User className="h-5 w-5 text-primary" />
                  Personal Information
                </CardTitle>
                <CardDescription>Please provide your basic details</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="name">Full Name</Label>
                    <Input
                      id="name"
                      placeholder="John Doe"
                      value={formData.name}
                      onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="age">Age</Label>
                    <Input
                      id="age"
                      type="number"
                      placeholder="25"
                      value={formData.age}
                      onChange={(e) => setFormData({ ...formData, age: e.target.value })}
                    />
                  </div>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="gender">Gender</Label>
                    <Select value={formData.gender} onValueChange={(value) => setFormData({ ...formData, gender: value })}>
                      <SelectTrigger id="gender">
                        <SelectValue placeholder="Select gender" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="male">Male</SelectItem>
                        <SelectItem value="female">Female</SelectItem>
                        <SelectItem value="other">Other</SelectItem>
                        <SelectItem value="prefer-not-to-say">Prefer not to say</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="area">Area Infected</Label>
                    <Input
                      id="area"
                      placeholder="e.g., Left arm, Back, Face"
                      value={formData.areaInfected}
                      onChange={(e) => setFormData({ ...formData, areaInfected: e.target.value })}
                    />
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        )}

        {currentStep === 2 && (
          <motion.div
            key="step2"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
          >
            <Card className="shadow-elevated border-border">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Upload className="h-5 w-5 text-primary" />
                  Upload Image
                </CardTitle>
                <CardDescription>Upload a clear image of the affected area</CardDescription>
              </CardHeader>
              <CardContent>
                <ImageUpload onImageUpload={handleImageUpload} currentImage={formData.imagePreview} />
              </CardContent>
            </Card>
          </motion.div>
        )}

        {currentStep === 3 && (
          <motion.div
            key="step3"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
          >
            <Card className="shadow-elevated border-border">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <FileText className="h-5 w-5 text-primary" />
                  Symptoms & Side Effects
                </CardTitle>
                <CardDescription>Select all symptoms you are experiencing</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {SIDE_EFFECTS.map((effect) => (
                    <div key={effect} className="flex items-center space-x-2">
                      <Checkbox
                        id={effect}
                        checked={formData.sideEffects.includes(effect)}
                        onCheckedChange={() => handleSideEffectToggle(effect)}
                      />
                      <label
                        htmlFor={effect}
                        className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70 cursor-pointer"
                      >
                        {effect}
                      </label>
                    </div>
                  ))}

                  <div className="flex items-start space-x-2">
                    <Checkbox
                      id="others"
                      checked={showOtherInput}
                      onCheckedChange={handleOthersToggle}
                    />
                    <div className="flex flex-col gap-2 w-full">
                      <label
                        htmlFor="others"
                        className="text-sm font-medium cursor-pointer"
                      >
                        Others (specify)
                      </label>
                      {showOtherInput && (
                        <Input
                          id="other-input"
                          placeholder="Enter additional symptom"
                          value={otherSymptom}
                          onChange={(e) => handleOtherInputChange(e.target.value)}
                        />
                      )}
                    </div>
                  </div>

                </div>
              </CardContent>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Navigation Buttons */}
      <div className="flex justify-between mt-6">
        <Button
          variant="outline"
          onClick={prevStep}
          disabled={currentStep === 1 || isAnalyzing}
          className="gap-2"
        >
          <ArrowLeft className="h-4 w-4" />
          Previous
        </Button>

        {currentStep < 3 ? (
          <Button onClick={nextStep} className="gap-2">
            Next
            <ArrowRight className="h-4 w-4" />
          </Button>
        ) : (
          <Button onClick={handleSubmit} disabled={isAnalyzing} className="gap-2 bg-gradient-to-r from-primary to-secondary">
            {isAnalyzing ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                Analyzing...
              </>
            ) : (
              <>
                Analyze
                <ArrowRight className="h-4 w-4" />
              </>
            )}
          </Button>
        )}
      </div>
    </div>
  );
};

export default DiagnosticForm;
