(this.webpackJsonpstreamlit_component_template=this.webpackJsonpstreamlit_component_template||[]).push([[0],{17:function(t,e,o){t.exports=o(28)},28:function(t,e,o){"use strict";o.r(e);var n=o(7),a=o.n(n),r=o(14),c=o.n(r),s=o(4),i=o.n(s),l=o(5),u=o(0),d=o(3),p=o(1),m=o(2),b=o(12),f=o(16),h=function(t){Object(p.a)(o,t);var e=Object(m.a)(o);function o(t){var n;return Object(u.a)(this,o),(n=e.call(this,t)).render=function(){n.props.args.name;var t=n.props.theme,e={};if(t){var o="1px solid ".concat(n.state.isFocused?t.primaryColor:"gray");e.border=o,e.outline=o}return a.a.createElement("div",null,a.a.createElement(f.ReactMic,{record:n.state.record,className:"sound-wave",mimeType:"audio/wav",onStop:n.onStop,onData:n.onData,channelCount:1,strokeColor:"#000000",backgroundColor:"#FFFFFF"}),a.a.createElement("button",{onClick:n.startRecording,type:"button"},"Start"),a.a.createElement("button",{onClick:n.stopRecording,type:"button"},"Stop"))},n.startRecording=function(){n.setState({record:!0})},n.stopRecording=function(){n.setState({record:!1})},n.state={record:!1,numClicks:0,isFocused:!1},n}return Object(d.a)(o,[{key:"onData",value:function(t){console.log("chunk of real-time data is: ",t)}},{key:"onStop",value:function(){var t=Object(l.a)(i.a.mark((function t(e){var o,n,a;return i.a.wrap((function(t){for(;;)switch(t.prev=t.next){case 0:console.log("recordedlob is: ",e),b.a.setComponentValue(e),(o=new XMLHttpRequest).onload=function(t){4===this.readyState&&console.log("Server returned: ",t.target.responseText)},(n=new FormData).append("audio_data",e.blob,"test.wav"),a=window.location.hostname.includes("localhost")||window.location.hostname.includes("127.0.0.1")?"http://localhost:5000":"",o.open("POST",a+"/api/save_audio",!0),o.send(n);case 9:case"end":return t.stop()}}),t)})));return function(e){return t.apply(this,arguments)}}()}]),o}(b.b),v=Object(b.c)(h);c.a.render(a.a.createElement(a.a.StrictMode,null,a.a.createElement(v,null)),document.getElementById("root"))}},[[17,1,2]]]);
//# sourceMappingURL=main.1b743b48.chunk.js.map