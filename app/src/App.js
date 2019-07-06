import React, { Component } from 'react';
import './App.css';
import { makeStyles } from '@material-ui/core/styles';
import ButtonAppBar from "./components/AppBar";
import CenteredTabs from './components/Tabs';


// to parse input file
// data = data.split('\n')
// data.slice(0,data.length - 1).map(e => JSON.parse(e))



const useStyles = makeStyles({
  nav: {
    flexGrow: 1,
    width:840,
    marginTop: 20,
    
  },
  label: {
    color: '#009BE5',
    fontWeight:700,
    fontSize:16,
},
indicator: {
    backgroundColor: '#009BE5'
  }
});


class App extends Component {

  render() {
    return(
      <div className='App'>
          <ButtonAppBar />
          <CenteredTabs />
          
      </div>
      
      )
    }
  }
  
  export default App;
  
  
  {/* <Header />
 <div className="strip"></div>
 <Router>
 <div className="App">
   <div className="">

   </div>
         <Route exact path='/' component={Container} />
         <Route exact path='/map' component={Map}/>
 </div>
 <ul className="router-links">
   <li className="link-stream"><Link to="/">Stream</Link></li>
   <li className="link-map"><Link to="/map">Map</Link></li>

 </ul>
 </Router> */}