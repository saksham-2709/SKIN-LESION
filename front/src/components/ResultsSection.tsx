import { motion } from "framer-motion";
import { AlertCircle, CheckCircle, FileDown, RefreshCw, TrendingUp } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { useToast } from "@/hooks/use-toast";

interface ResultsSectionProps {
  data: any;
  onNewDiagnosis: () => void;
}

const ResultsSection = ({ data, onNewDiagnosis }: ResultsSectionProps) => {
  const { toast } = useToast();

  const handleExportPDF = async () => {
    try {
      toast({
        title: "Export initiated",
        description: "Your health report is being generated...",
      });

      // API endpoint - default to localhost:5000, can be configured via env
      const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';
      
      const response = await fetch(`${API_URL}/report`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
      });

      if (!response.ok) {
        throw new Error('Failed to generate PDF');
      }

      // Get the PDF blob and download it
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'derm_insight_report.pdf';
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);

      toast({
        title: "Report ready",
        description: "Your health report has been downloaded",
      });
    } catch (error: any) {
      console.error('PDF export error:', error);
      toast({
        title: "Export failed",
        description: error.message || "Failed to generate PDF. Please try again.",
        variant: "destructive",
      });
    }
  };

  const confidenceColor = data.confidence >= 80 ? "text-green-600" : data.confidence >= 60 ? "text-yellow-600" : "text-orange-600";

  return (
    <div className="space-y-6">
      {/* Action Buttons */}
      <div className="flex flex-wrap gap-3 justify-between items-center">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <h2 className="text-2xl font-bold text-foreground">Analysis Results</h2>
          <p className="text-sm text-muted-foreground">Based on AI analysis of your submitted image</p>
        </motion.div>
        <div className="flex gap-2">
          <Button variant="outline" onClick={onNewDiagnosis} className="gap-2">
            <RefreshCw className="h-4 w-4" />
            New Analysis
          </Button>
          <Button onClick={handleExportPDF} className="gap-2 bg-gradient-to-r from-primary to-secondary">
            <FileDown className="h-4 w-4" />
            Export PDF
          </Button>
        </div>
      </div>

      {/* Main Diagnosis Card */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        <Card className="shadow-elevated border-border">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5 text-primary" />
              Diagnosis
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-start justify-between">
              <div>
                <h3 className="text-2xl font-bold text-foreground mb-2">{data.diagnosis}</h3>
                <p className="text-sm text-muted-foreground">
                  Confidence: <span className={`font-semibold ${confidenceColor}`}>{data.confidence}%</span>
                </p>
              </div>
              <Badge variant={data.confidence >= 80 ? "default" : "secondary"} className="text-sm">
                {data.confidence >= 80 ? "High Confidence" : "Moderate Confidence"}
              </Badge>
            </div>

            <Separator />

            {/* Image Comparison */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <p className="text-sm font-semibold mb-2 text-foreground">Original Image</p>
                <img
                  src={data.imagePreview}
                  alt="Original"
                  className="w-full h-48 object-cover rounded-lg border border-border"
                />
              </div>
              <div>
                <p className="text-sm font-semibold mb-2 text-foreground">Grad-CAM Heatmap</p>
                <img
                  src={data.gradCamUrl}
                  alt="Grad-CAM"
                  className="w-full h-48 object-cover rounded-lg border border-border"
                />
              </div>
            </div>

            {/* Patient Info */}
            <div className="bg-muted/50 rounded-lg p-4">
              <h4 className="font-semibold mb-3 text-foreground">Patient Information</h4>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div>
                  <p className="text-muted-foreground">Name</p>
                  <p className="font-medium text-foreground">{data.name}</p>
                </div>
                <div>
                  <p className="text-muted-foreground">Age</p>
                  <p className="font-medium text-foreground">{data.age}</p>
                </div>
                <div>
                  <p className="text-muted-foreground">Gender</p>
                  <p className="font-medium text-foreground capitalize">{data.gender}</p>
                </div>
                <div>
                  <p className="text-muted-foreground">Area Affected</p>
                  <p className="font-medium text-foreground">{data.areaInfected}</p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* Causes */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
      >
        <Card className="shadow-card border-border">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <AlertCircle className="h-5 w-5 text-primary" />
              Possible Causes
            </CardTitle>
            <CardDescription>Common factors that may contribute to this condition</CardDescription>
          </CardHeader>
          <CardContent>
            <ul className="space-y-2">
              {data.causes.map((cause: string, index: number) => (
                <li key={index} className="flex items-start gap-2">
                  <div className="mt-1 h-1.5 w-1.5 rounded-full bg-primary flex-shrink-0" />
                  <span className="text-sm text-foreground">{cause}</span>
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      </motion.div>

      {/* Home Remedies */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
      >
        <Card className="shadow-card border-border">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <CheckCircle className="h-5 w-5 text-secondary" />
              Home Care Recommendations
            </CardTitle>
            <CardDescription>General care tips (Always consult a dermatologist)</CardDescription>
          </CardHeader>
          <CardContent>
            <ul className="space-y-2">
              {data.remedies.map((remedy: string, index: number) => (
                <li key={index} className="flex items-start gap-2">
                  <CheckCircle className="h-4 w-4 mt-0.5 text-secondary flex-shrink-0" />
                  <span className="text-sm text-foreground">{remedy}</span>
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      </motion.div>

      {/* Side Effects & Symptoms */}
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4 }}>
        <Card className="shadow-card border-border">
          <CardHeader>
            <CardTitle>Reported & Related Symptoms</CardTitle>
            <CardDescription>
              Includes both listed and custom symptoms you entered
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {/* Reported Symptoms */}
              <div>
                <p className="text-sm font-semibold mb-2 text-foreground">
                  Reported Symptoms:
                </p>
                <div className="flex flex-wrap gap-2">
                  {data.sideEffects?.length ? (
                    data.sideEffects.map((effect: string, index: number) => (
                      <Badge
                        key={index}
                        variant="outline"
                        className={`${
                          effect.trim().toLowerCase() !==
                            "itching" &&
                          effect.trim().toLowerCase() !== "pain" &&
                          effect.trim().toLowerCase() !== "swelling" &&
                          effect.trim().toLowerCase() !== "redness"
                            ? "border-primary text-primary"
                            : ""
                        }`}
                      >
                        {effect}
                      </Badge>
                    ))
                  ) : (
                    <p className="text-muted-foreground text-sm">
                      No symptoms reported
                    </p>
                  )}
                </div>
              </div>

              <Separator />

              {/* Additional Effects */}
              <div>
                <p className="text-sm font-semibold mb-2 text-foreground">
                  Possible Additional Effects:
                </p>
                <ul className="space-y-1">
                  {data.relatedSideEffects?.length ? (
                    data.relatedSideEffects.map((effect: string, index: number) => (
                      <li key={index} className="flex items-start gap-2">
                        <div className="mt-1.5 h-1 w-1 rounded-full bg-muted-foreground flex-shrink-0" />
                        <span className="text-sm text-muted-foreground">{effect}</span>
                      </li>
                    ))
                  ) : (
                    <p className="text-muted-foreground text-sm">
                      No related side effects provided
                    </p>
                  )}
                </ul>
              </div>
            </div>
          </CardContent>
        </Card>
      </motion.div>
    </div>
  );
};

export default ResultsSection;