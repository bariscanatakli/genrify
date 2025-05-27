/**
 * Extract JSON from mixed output with markers or by regexp matching
 * This helps parse Python output that contains logs, warnings, and JSON
 */
export function extractJsonFromOutput(stdout: string): any {
  // Look for marked JSON sections
  const jsonStartMarker = "===JSON_START===";
  const jsonEndMarker = "===JSON_END===";
  
  const startIdx = stdout.indexOf(jsonStartMarker);
  if (startIdx >= 0) {
    const contentStart = startIdx + jsonStartMarker.length;
    const endIdx = stdout.indexOf(jsonEndMarker, contentStart);
    
    if (endIdx >= 0) {
      const jsonStr = stdout.substring(contentStart, endIdx).trim();
      try {
        return JSON.parse(jsonStr);
      } catch (e) {
        console.error("JSON parse error in marked section:", e);
        console.error("Raw JSON string:", jsonStr);
      }
    }
  }
  
  // Fallback: try to find JSON object or array
  try {
    // Try to find a JSON object
    const objMatch = stdout.match(/({[\s\S]*})/);
    if (objMatch && objMatch[0]) {
      return JSON.parse(objMatch[0]);
    }
    
    // Try to find a JSON array
    const arrMatch = stdout.match(/(\[[\s\S]*\])/);
    if (arrMatch && arrMatch[0]) {
      return JSON.parse(arrMatch[0]);
    }
  } catch (e) {
    console.error("Fallback JSON extraction failed:", e);
  }
  
  // If all parsing attempts fail, throw an error
  throw new Error("Could not extract valid JSON from output");
}

/**
 * Create a Python-safe command to output JSON with markers
 */
export function createPythonJsonCommand(
  scriptPath: string, 
  functionName: string, 
  filePath: string, 
  ...args: any[]
): string {
  // Format arguments as Python literals
  const pyArgs = args.map(arg => {
    if (typeof arg === 'string') return `'${arg}'`;
    if (typeof arg === 'number') return arg.toString();
    if (typeof arg === 'boolean') return arg ? 'True' : 'False';
    return JSON.stringify(arg);
  }).join(', ');
  
  return `python -c "import sys; sys.path.append('${scriptPath}'); from process import ${functionName}, output_json; output_json(${functionName}('${filePath}'${args.length ? ', ' + pyArgs : ''}))" 2>&1`;
}
