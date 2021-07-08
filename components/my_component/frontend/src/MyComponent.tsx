import {
  Streamlit,
  StreamlitComponentBase,
  withStreamlitConnection,
} from "streamlit-component-lib"
import React, { ReactNode } from "react"
// Import react-mic
import { ReactMic } from 'react-mic';

interface State {
  record: boolean
  numClicks: number
  isFocused: boolean
}

/**
 * This is a React-based component template. The `render()` function is called
 * automatically when your component should be re-rendered.
 */
class MyComponent extends StreamlitComponentBase<State> {
  constructor(props: any) {
    super(props);
    this.state = {
      record: false,
      numClicks: 0,
      isFocused: false
    }
  }
  
  // public state = { numClicks: 0, isFocused: false }

  public render = (): ReactNode => {
    // Arguments that are passed to the plugin in Python are accessible
    // via `this.props.args`. Here, we access the "name" arg.
    const name = this.props.args["name"]

    // Streamlit sends us a theme object via props that we can use to ensure
    // that our component has visuals that match the active theme in a
    // streamlit app.
    const { theme } = this.props
    const style: React.CSSProperties = {}

    // Maintain compatibility with older versions of Streamlit that don't send
    // a theme object.
    if (theme) {
      // Use the theme object to style our button border. Alternatively, the
      // theme style is defined in CSS vars.
      const borderStyling = `1px solid ${
        this.state.isFocused ? theme.primaryColor : "gray"
      }`
      style.border = borderStyling
      style.outline = borderStyling
    }

    // Show a button and some text.
    // When the button is clicked, we'll increment our "numClicks" state
    // variable, and send its new value back to Streamlit, where it'll
    // be available to the Python program.
    return (
      <div>
        <ReactMic
          record={this.state.record}
          className="sound-wave"
          mimeType="audio/wav"
          onStop={this.onStop}
          onData={this.onData}
          channelCount={1}
          strokeColor="#000000"
          backgroundColor="#FFFFFF" />
        <button onClick={this.startRecording} type="button">Start</button>
        <button onClick={this.stopRecording} type="button">Stop</button>
      </div>

    )
  }

  // React-mic
  startRecording = () => {
    this.setState({ record: true });
  }
 
  stopRecording = () => {
    this.setState(
      { record: false },
    );
  }
 
  onData(recordedBlob: any) {
    console.log('chunk of real-time data is: ', recordedBlob);
  }
 
  async onStop(recordedBlob: any) {
    console.log('recordedlob is: ', recordedBlob);
    
    Streamlit.setComponentValue(recordedBlob)

    var xhr=new XMLHttpRequest();
    xhr.onload=function(e: any) {
        if(this.readyState === 4) {
            console.log("Server returned: ",e.target.responseText);
        }
    };
    var fd=new FormData();
    fd.append("audio_data",recordedBlob.blob, 'test.wav');
    const prefix = window.location.hostname.includes("localhost") || window.location.hostname.includes("127.0.0.1") ? "http://localhost:5000" : "";
    xhr.open("POST",prefix + "/api/save_audio",true);
    xhr.send(fd);
  }
}

// "withStreamlitConnection" is a wrapper function. It bootstraps the
// connection between your component and the Streamlit app, and handles
// passing arguments from Python -> Component.
//
// You don't need to edit withStreamlitConnection (but you're welcome to!).
export default withStreamlitConnection(MyComponent)
