'''import logging
import speech_recognition as sr
from pydub import AudioSegment
from io import BytesIO
import os
from groq import Groq

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Explicitly set ffmpeg path
ffmpeg_path = "C:\\ffmpeg\\bin\\ffmpeg.exe"  
AudioSegment.converter = ffmpeg_path
logging.info(f"Using ffmpeg at: {ffmpeg_path}")

def record_audio(file_path, timeout=15, phrase_time_limit=None):
    recognizer = sr.Recognizer()
    audio_data = None
    
    try:
        with sr.Microphone() as source:
            logging.info("Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            logging.info("Start speaking now...")

            # Recording the audio
            audio_data = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            logging.info("Recording complete")

            # Saving the audio
            if audio_data:
                wav_data = audio_data.get_wav_data()
                audio_segment = AudioSegment.from_wav(BytesIO(wav_data))
                audio_segment.export(file_path, format="mp3", bitrate="128k")
                logging.info(f"Audio saved to {file_path}")
                return True
            else:
                logging.error("No audio data captured")
                return False

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return False

def transcribe_audio(file_path):
    try:
        # Check if file exists and has content
        if not os.path.exists(file_path):
            logging.error(f"Audio file not found: {file_path}")
            return None
            
        if os.path.getsize(file_path) == 0:
            logging.error(f"Audio file is empty: {file_path}")
            return None
            
        # Set up STT model
        GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
        if not GROQ_API_KEY:
            logging.error("GROQ_API_KEY environment variable not set")
            return None
            
        client = Groq(api_key=GROQ_API_KEY)
        stt_model = "whisper-large-v3"
        
        with open(file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model=stt_model,
                file=audio_file,
                language="en"
            )
            return transcription.text
            
    except Exception as e:
        logging.error(f"Transcription error: {e}")
        return None

# Main execution
if __name__ == "__main__":
    audio_filepath = "./voice_data.mp3"
    
    # Record audio
    recording_success = record_audio(file_path=audio_filepath)
    
    # Transcribe if recording was successful
    if recording_success:
        transcription = transcribe_audio(audio_filepath)
        if transcription:
            print("Transcription result:")
            print(transcription)
        else:
            print("Transcription failed or returned empty result")
    else:
        print("Audio recording failed, cannot proceed with transcription")'''
import logging
import os
import json
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import language_tool_python
import speech_recognition as sr
from pydub import AudioSegment
from io import BytesIO
from groq import Groq

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configure paths
ffmpeg_path = "C:\\ffmpeg\\bin\\ffmpeg.exe"  # Update with your path
AudioSegment.converter = ffmpeg_path
logging.info(f"Using ffmpeg at: {ffmpeg_path}")

# Download necessary NLTK resources
try:
    nltk.download('punkt')
    nltk.download('stopwords')
except Exception as e:
    logging.warning(f"NLTK download error: {e}")

class VoiceRecorder:
    def __init__(self, timeout=15, phrase_time_limit=None):
        self.timeout = timeout
        self.phrase_time_limit = phrase_time_limit
        
    def record_audio(self, file_path):
        recognizer = sr.Recognizer()
        audio_data = None
        
        try:
            with sr.Microphone() as source:
                logging.info("Adjusting for ambient noise...")
                recognizer.adjust_for_ambient_noise(source, duration=1)
                logging.info("Start speaking now...")

                # Recording the audio
                audio_data = recognizer.listen(source, timeout=self.timeout, phrase_time_limit=self.phrase_time_limit)
                logging.info("Recording complete")

                # Saving the audio
                if audio_data:
                    wav_data = audio_data.get_wav_data()
                    audio_segment = AudioSegment.from_wav(BytesIO(wav_data))
                    audio_segment.export(file_path, format="mp3", bitrate="128k")
                    logging.info(f"Audio saved to {file_path}")
                    return True
                else:
                    logging.error("No audio data captured")
                    return False

        except Exception as e:
            logging.error(f"An error occurred: {e}")
            return False

class Transcriber:
    def __init__(self, api_key=None, model="whisper-large-v3"):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            logging.warning("GROQ_API_KEY not set. Transcription will fail.")
        self.model = model
        
    def transcribe(self, file_path):
        try:
            # Check if file exists and has content
            if not os.path.exists(file_path):
                logging.error(f"Audio file not found: {file_path}")
                return None
                
            if os.path.getsize(file_path) == 0:
                logging.error(f"Audio file is empty: {file_path}")
                return None
                
            # Set up STT model
            client = Groq(api_key=self.api_key)
            
            with open(file_path, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    model=self.model,
                    file=audio_file,
                    language="en"
                )
                return transcription.text
                
        except Exception as e:
            logging.error(f"Transcription error: {e}")
            return None

class GrammarScorer:
    def __init__(self):
        self.tool = language_tool_python.LanguageTool('en-US')
        self.stop_words = set(stopwords.words('english'))
        
    def analyze_text(self, text):
        """Analyze text and return grammar metrics"""
        if not text:
            return None
            
        # Basic text metrics
        word_count = len(word_tokenize(text))
        sentence_count = max(1, len(sent_tokenize(text)))
        avg_sentence_length = word_count / sentence_count
        
        # Grammar error detection
        matches = self.tool.check(text)
        grammar_errors = len(matches)
        error_rate = grammar_errors / word_count if word_count > 0 else 0
        
        # Vocabulary diversity
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalpha()]
        content_words = [w for w in words if w not in self.stop_words]
        unique_words = len(set(content_words))
        vocabulary_diversity = unique_words / len(content_words) if content_words else 0
        
        # Calculate grammar score (0-100)
        error_penalty = min(50, 50 * error_rate * sentence_count)
        base_score = 100 - error_penalty
        
        # Adjust score based on vocabulary diversity and sentence complexity
        diversity_bonus = vocabulary_diversity * 10  # 0-10 points
        complexity_factor = min(10, abs(avg_sentence_length - 15)) / 2  # Penalize too simple or complex
        complexity_adjustment = 5 - complexity_factor  # -5 to 5 points
        
        final_score = min(100, max(0, base_score + diversity_bonus + complexity_adjustment))
        
        return {
            "text": text,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_sentence_length": avg_sentence_length,
            "grammar_errors": grammar_errors,
            "error_rate": error_rate,
            "vocabulary_diversity": vocabulary_diversity,
            "grammar_score": round(final_score, 1),
            "error_details": [{"rule_id": match.ruleId, 
                              "message": match.message,
                              "context": text[max(0, match.offset-20):min(len(text), match.offset+match.errorLength+20)]} 
                             for match in matches[:10]]  # Limit to top 10 errors
        }
        
    def get_suggestions(self, text):
        """Get detailed correction suggestions"""
        matches = self.tool.check(text)
        suggestions = []
        
        for match in matches:
            suggestions.append({
                "error": text[match.offset:match.offset+match.errorLength],
                "suggestions": match.replacements[:3],  # Top 3 suggestions
                "explanation": match.message,
                "context": text[max(0, match.offset-20):min(len(text), match.offset+match.errorLength+20)]
            })
            
        return suggestions

class GrammarScoringEngine:
    def __init__(self, output_dir="./results"):
        self.recorder = VoiceRecorder()
        self.transcriber = Transcriber()
        self.scorer = GrammarScorer()
        self.output_dir = output_dir
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
    def process_audio(self, audio_path=None):
        """Process audio file or record new audio"""
        if not audio_path:
            audio_path = os.path.join(self.output_dir, "voice_sample.mp3")
            recording_success = self.recorder.record_audio(audio_path)
            if not recording_success:
                return {"error": "Recording failed"}
                
        # Transcribe audio
        text = self.transcriber.transcribe(audio_path)
        if not text:
            return {"error": "Transcription failed or returned empty result"}
            
        # Score grammar
        analysis = self.scorer.analyze_text(text)
        if not analysis:
            return {"error": "Grammar analysis failed"}
            
        # Get improvement suggestions
        suggestions = self.scorer.get_suggestions(text)
        analysis["improvement_suggestions"] = suggestions
        
        # Save results
        result_path = os.path.join(self.output_dir, "grammar_analysis.json")
        with open(result_path, 'w') as f:
            json.dump(analysis, f, indent=2)
            
        logging.info(f"Grammar analysis saved to {result_path}")
        return analysis
        
    def process_audio_batch(self, audio_dir):
        """Process multiple audio files from a directory"""
        results = []
        
        for filename in os.listdir(audio_dir):
            if filename.endswith(('.mp3', '.wav', '.m4a')):
                file_path = os.path.join(audio_dir, filename)
                logging.info(f"Processing {file_path}")
                
                result = self.process_audio(file_path)
                result["filename"] = filename
                results.append(result)
                
        # Create a DataFrame with results
        df = pd.DataFrame(results)
        csv_path = os.path.join(self.output_dir, "batch_results.csv")
        df.to_csv(csv_path, index=False)
        
        return results

# For Kaggle Notebook usage
class KaggleGrammarScorer:
    def __init__(self):
        self.engine = GrammarScoringEngine()
        
    def setup_kaggle_display(self):
        """Setup display functions for Kaggle notebook"""
        try:
            from IPython.display import display, HTML
            import matplotlib.pyplot as plt
            
            def display_result(result):
                """Display grammar scoring results in a nice format"""
                if "error" in result:
                    display(HTML(f"<div style='color:red'><h3>Error</h3><p>{result['error']}</p></div>"))
                    return
                    
                html = f"""
                <div style='background:#f8f9fa;padding:20px;border-radius:5px'>
                    <h2>Grammar Analysis Results</h2>
                    <p><b>Transcribed Text:</b> {result['text']}</p>
                    <h3>Grammar Score: <span style='color:{"green" if result["grammar_score"] >= 80 else "orange" if result["grammar_score"] >= 60 else "red"}'>{result["grammar_score"]}/100</span></h3>
                    
                    <h4>Stats:</h4>
                    <ul>
                        <li>Word count: {result['word_count']}</li>
                        <li>Sentence count: {result['sentence_count']}</li>
                        <li>Average sentence length: {result['avg_sentence_length']:.1f} words</li>
                        <li>Grammar errors: {result['grammar_errors']}</li>
                        <li>Error rate: {result['error_rate']:.3f} errors per word</li>
                        <li>Vocabulary diversity: {result['vocabulary_diversity']:.3f}</li>
                    </ul>
                    
                    <h4>Error Details:</h4>
                    <ul>
                """
                
                for error in result.get('error_details', []):
                    html += f"""
                        <li>
                            <b>{error['rule_id']}</b>: {error['message']}<br>
                            <i>Context: "...{error['context']}..."</i>
                        </li>
                    """
                
                html += """
                    </ul>
                    
                    <h4>Improvement Suggestions:</h4>
                    <ul>
                """
                
                for suggestion in result.get('improvement_suggestions', []):
                    replacements = ", ".join(suggestion['suggestions']) if suggestion['suggestions'] else "No specific suggestions"
                    html += f"""
                        <li>
                            "<b>{suggestion['error']}</b>" could be replaced with: {replacements}<br>
                            <i>Explanation: {suggestion['explanation']}</i>
                        </li>
                    """
                
                html += """
                    </ul>
                </div>
                """
                
                display(HTML(html))
                
                # Plot a radar chart for visualization
                categories = ['Grammar', 'Vocabulary', 'Complexity']
                grammar_score = min(1, 1 - result['error_rate'] * 3)
                vocabulary_score = result['vocabulary_diversity']
                complexity_score = min(1, 1 - abs(result['avg_sentence_length'] - 15) / 15)
                
                values = [grammar_score, vocabulary_score, complexity_score]
                
                # Create the radar chart
                angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
                values += values[:1]
                angles += angles[:1]
                categories += categories[:1]
                
                fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
                ax.fill(angles, values, color='skyblue', alpha=0.25)
                ax.plot(angles, values, color='blue', linewidth=2)
                
                ax.set_yticklabels([])
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(categories[:-1])
                
                for angle, value, category in zip(angles[:-1], values[:-1], categories[:-1]):
                    ax.text(angle, value + 0.1, f"{category}: {value:.2f}", 
                            horizontalalignment='center', size=10)
                
                plt.title('Language Skills Assessment', size=15)
                plt.tight_layout()
                plt.show()
                
            self.display_result = display_result
            return True
            
        except ImportError:
            logging.warning("Could not set up Kaggle display - missing dependencies")
            return False

# Example usage
if __name__ == "__main__":
    engine = GrammarScoringEngine()
    
    # Process a single audio recording
    result = engine.process_audio()
    
    if "error" not in result:
        print(f"Grammar Score: {result['grammar_score']}/100")
        print(f"Words: {result['word_count']}, Sentences: {result['sentence_count']}")
        print(f"Grammar errors: {result['grammar_errors']}")
        
        print("\nSuggestions for improvement:")
        for suggestion in result.get('improvement_suggestions', [])[:5]:  # Show top 5
            print(f"- {suggestion['error']} -> {suggestion['suggestions']}")
    else:
        print(f"Error: {result['error']}")