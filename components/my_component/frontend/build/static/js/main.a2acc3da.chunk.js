(this.webpackJsonpstreamlit_component_template=this.webpackJsonpstreamlit_component_template||[]).push([[0],{17:function(e,t,n){e.exports=n(28)},28:function(e,t,n){"use strict";n.r(t);var o=n(7),a=n.n(o),r=n(14),c=n.n(r),s=n(4),i=n.n(s),u=n(5),l=n(0),p=n(3),d=n(1),m=n(2),b=n(12),f=n(16),v=function(e){Object(d.a)(n,e);var t=Object(m.a)(n);function n(e){var o;return Object(l.a)(this,n),(o=t.call(this,e)).render=function(){o.props.args.name;var e=o.props.theme,t={};if(e){var n="1px solid ".concat(o.state.isFocused?e.primaryColor:"gray");t.border=n,t.outline=n}return a.a.createElement("div",null,a.a.createElement(f.ReactMic,{record:o.state.record,className:"sound-wave",mimeType:"audio/wav",onStop:o.onStop,onData:o.onData,channelCount:1,strokeColor:"#000000",backgroundColor:"#FFFFFF"}),a.a.createElement("button",{onClick:o.startRecording,type:"button"},"Start"),a.a.createElement("button",{onClick:o.stopRecording,type:"button"},"Stop"))},o.startRecording=function(){o.setState({record:!0})},o.stopRecording=function(){o.setState({record:!1})},o.state={record:!1,numClicks:0,isFocused:!1},o}return Object(p.a)(n,[{key:"onData",value:function(e){console.log("chunk of real-time data is: ",e)}},{key:"onStop",value:function(){var e=Object(u.a)(i.a.mark((function e(t){var n,o;return i.a.wrap((function(e){for(;;)switch(e.prev=e.next){case 0:console.log("recordedlob is: ",t),b.a.setComponentValue(t),(n=new XMLHttpRequest).onload=function(e){4===this.readyState&&console.log("Server returned: ",e.target.responseText)},(o=new FormData).append("audio_data",t.blob,"test.wav"),n.open("POST","/api/save_audio",!0),n.send(o);case 8:case"end":return e.stop()}}),e)})));return function(t){return e.apply(this,arguments)}}()}]),n}(b.b),g=Object(b.c)(v);c.a.render(a.a.createElement(a.a.StrictMode,null,a.a.createElement(g,null)),document.getElementById("root"))}},[[17,1,2]]]);
//# sourceMappingURL=main.a2acc3da.chunk.js.map