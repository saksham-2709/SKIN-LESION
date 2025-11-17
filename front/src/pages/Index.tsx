import { useState } from "react";
import { motion } from "framer-motion";
import { Activity } from "lucide-react";
import DiagnosticForm from "@/components/DiagnosticForm";
import ResultsSection from "@/components/ResultsSection";
import DermatologistFinder from "@/components/DermatologistFinder";

const Index = () => {
  const [step, setStep] = useState<"form" | "results">("form");
  const [diagnosticData, setDiagnosticData] = useState<any>(null);

  const handleDiagnosisComplete = (data: any) => {
    setDiagnosticData(data);
    setStep("results");
  };

  const handleNewDiagnosis = () => {
    setStep("form");
    setDiagnosticData(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-accent/5 to-background">
      {/* Header */}
      <header className="border-b border-border bg-card/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4">
          <motion.div 
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex items-center gap-3"
          >
            <div className="p-2 rounded-xl bg-gradient-to-br from-primary to-secondary">
              <Activity className="h-6 w-6 text-primary-foreground" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-foreground">DermaScan AI</h1>
              <p className="text-sm text-muted-foreground">Advanced Skin Anomaly Detection</p>
            </div>
          </motion.div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8">
        {step === "form" ? (
          <DiagnosticForm onComplete={handleDiagnosisComplete} />
        ) : (
          <div className="space-y-6">
            <ResultsSection data={diagnosticData} onNewDiagnosis={handleNewDiagnosis} />
            <DermatologistFinder />
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-border mt-16 py-8 bg-card/50">
        <div className="container mx-auto px-4 text-center text-sm text-muted-foreground">
          <p className="mb-2">
            <strong>Disclaimer:</strong> This tool is for educational purposes only and should not replace professional medical advice.
          </p>
          <p>Always consult with a qualified dermatologist for accurate diagnosis and treatment.</p>
        </div>
      </footer>
    </div>
  );
};

export default Index;
