import os
import logging
from flask import Flask, render_template, request, redirect, url_for, flash
from dotenv import load_dotenv
import markdown # For rendering the report
import socialosintlm # Your core logic script
import argparse # To create a dummy args object if needed
from datetime import datetime
import json # For potential future structured data handling
from flask import Response

# --- Basic Setup ---
load_dotenv()
app = Flask(__name__)
# SECRET_KEY is required for flashing messages
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'a_default_secret_key_change_me')

# --- Logging Setup (Optional but Recommended) ---
# Keep logging similar to your original script
logging.basicConfig(
    level=logging.INFO, # Adjust level as needed (DEBUG, INFO, WARN)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('analyser_web.log'), logging.StreamHandler()]
)
logger = logging.getLogger('SocialOSINTLM_Web')

# --- Initialize Analyser ---
# Create a basic Namespace object if your SocialOSINTLM __init__ expects one.
# You might need to adjust this based on what args SocialOSINTLM *actually* uses internally.
# For now, we assume it mainly needs it for initialization paths or flags.
dummy_args = argparse.Namespace(
    # Add any attributes SocialOSINTLM's __init__ or methods might expect,
    # even if they are not used in the web context (set defaults).
    # Example: If _save_output used args.format internally:
    format='markdown',
    no_auto_save=True # Saving is handled explicitly in the web route
)

try:
    # It's good practice to initialize potentially heavy objects once
    analyser = socialosintlm.SocialOSINTLM(args=dummy_args)
    # Verify essential API keys are present early
    analyser._verify_env_vars() # Run the check
    logger.info("SocialOSINTLM Analyser initialized successfully.")
except RuntimeError as e:
    logger.critical(f"Failed to initialize SocialOSINTLM: {e}", exc_info=True)
    # If analyser fails to init, we can't really run the app.
    # In a real app, you might show a persistent error page.
    # For simplicity here, we'll let it potentially raise later,
    # but logging the critical error is important.
    analyser = None # Indicate failure
except Exception as e:
    logger.critical(f"Unexpected error during analyser initialization: {e}", exc_info=True)
    analyser = None

# --- Flask Routes ---
@app.route('/download_report', methods=['POST'])
def download_report():
    report_markdown = request.form.get('report_markdown')
    query = request.form.get('query', 'analysis') # Get query for filename
    if not report_markdown:
        flash("No report content available to download.", "error")
        # Redirect back to where they might have been, or index
        # This part is tricky without knowing the original report URL
        # Maybe disable download if markdown isn't present initially
        return redirect(url_for('index'))

    # Create a safe filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_query = "".join(c if c.isalnum() else '_' for c in query[:30]).strip('_')
    filename = f"analysis_report_{timestamp}_{safe_query}.md"

    return Response(
        report_markdown,
        mimetype="text/markdown",
        headers={"Content-disposition":
                 f"attachment; filename={filename}"}
    )

@app.route('/', methods=['GET', 'POST'])
def index():
    if analyser is None:
        flash("Critical Error: Analyser could not be initialized. Check logs and API keys.", "error")
        return render_template('index.html', available_platforms=[])

    available_platforms = analyser.get_available_platforms()

    if request.method == 'POST':
        platforms_to_query = {}
        form_has_input = False
        for platform in available_platforms:
            # Input name matches the loop variable (e.g., 'twitter_usernames')
            input_name = f'{platform}_usernames'
            usernames_str = request.form.get(input_name, '').strip()
            if usernames_str:
                # Split, strip whitespace, and filter out empty strings
                usernames = [u.strip() for u in usernames_str.split(',') if u.strip()]
                if usernames:
                    platforms_to_query[platform] = usernames
                    form_has_input = True

        query = request.form.get('query', '').strip()
        save_report_flag = request.form.get('save_report') == 'on'

        # --- Input Validation ---
        if not form_has_input:
            flash('Please enter at least one username for a selected platform.', 'warning')
            return redirect(url_for('index'))
        if not query:
            flash('Please enter an analysis query.', 'warning')
            return redirect(url_for('index'))

        logger.info(f"Received analysis request. Platforms: {list(platforms_to_query.keys())}, Query: '{query}', Save: {save_report_flag}")

        # --- Run Analysis ---
        try:
            # Use a loading indicator here in a real app (JS/AJAX)
            #flash('Analysis started... please wait.', 'info') # Basic feedback
            report_markdown = analyser.analyse(platforms_to_query, query)

            if not report_markdown or report_markdown.strip().startswith(("[red]", "[yellow]")):
                error_message = report_markdown if report_markdown else "Analysis returned no result."
                logger.warning(f"Analysis completed with issues: {error_message}")
                # Determine category based on prefix
                error_category = "error" if error_message.strip().startswith("[red]") else "warning"
                flash(f"Analysis {error_category.capitalize()}: {error_message}", error_category)
                # Render report page but signal it's an error state maybe?
                return render_template('report.html',
                                    query=query,
                                    report_html=f"<article class='flash-{error_category}'>{error_message}</article>", # Use flash styling
                                    report_markdown=None, # No valid markdown
                                    is_error_report=True) # Pass a flag

            # --- Render Markdown to HTML ---
            report_html = markdown.markdown(report_markdown, extensions=['fenced_code', 'tables', 'nl2br']) # nl2br for line breaks
            logger.info("Analysis successful, report generated.")

            # --- Handle Saving (if requested and successful) ---
            saved_path = None
            if save_report_flag:
                try:
                    saved_path = save_analysis_output(report_markdown, query, list(platforms_to_query.keys()))
                    flash(f"Report successfully saved to: {saved_path}", "success")
                    logger.info(f"Report saved to {saved_path}")
                except Exception as e:
                    logger.error(f"Failed to save report: {e}", exc_info=True)
                    flash(f"Report generated, but failed to save: {e}", "error")

            # --- Render Report Page ---
            return render_template('report.html',
                                   query=query,
                                   report_html=report_html,
                                   report_markdown=report_markdown) # Pass raw markdown if needed (e.g., for copy/download)

        except socialosintlm.RateLimitExceededError as rle:
             logger.error(f"Analysis failed due to rate limit: {rle}", exc_info=True)
             flash(f"Analysis Aborted: Rate Limit Exceeded. {rle}", "error")
             return redirect(url_for('index')) # Go back to form on critical errors
        except (socialosintlm.UserNotFoundError, socialosintlm.AccessForbiddenError) as afe:
             logger.error(f"Analysis failed due to access error: {afe}", exc_info=True)
             flash(f"Analysis Failed: User Not Found or Access Forbidden. {afe}", "error")
             return redirect(url_for('index'))
        except Exception as e:
            logger.error(f"Unexpected error during analysis: {e}", exc_info=True)
            flash(f"An unexpected error occurred during analysis: {e}", "error")
            return redirect(url_for('index'))

    # --- Handle GET Request ---
    return render_template('index.html', available_platforms=available_platforms)


# --- Helper Function for Saving ---
def save_analysis_output(content: str, query: str, platforms_analysed: list) -> str:
    """Saves the analysis report markdown to a file in data/outputs."""
    output_dir = os.path.join("data", "outputs")
    os.makedirs(output_dir, exist_ok=True) # Ensure directory exists

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_query = "".join(c if c.isalnum() else '_' for c in query[:30]).strip('_')
    safe_platforms = "_".join(sorted(platforms_analysed))[:20]
    filename = f"analysis_{timestamp}_{safe_platforms}_{safe_query}.md"
    filepath = os.path.join(output_dir, filename)

    # Add metadata as comments/frontmatter to markdown
    md_metadata = f"""---
Query: {query}
Platforms: {', '.join(platforms_analysed)}
Timestamp: {datetime.now().isoformat()}
Text Model: {os.getenv('ANALYSIS_MODEL', 'unknown')}
Image Model: {os.getenv('IMAGE_ANALYSIS_MODEL', 'unknown')}
---

"""
    full_content = md_metadata + content

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(full_content)

    return filepath # Return the path where it was saved


# --- Main Execution ---
if __name__ == '__main__':
    # Use host='0.0.0.0' to make it accessible on your network (optional)
    # debug=True is useful for development, but should be False in production
    app.run(debug=True, host='127.0.0.1', port=5000)