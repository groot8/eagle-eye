import React, { Component } from 'react';
import './App.css';
import Header from './Components/Header/Header';
import Sidebar from './Components/Sidebar/Sidebar';
import Container from './Components/Container/Container';

// to parse input file
// data = data.split('\n')
// data.slice(0,data.length - 1).map(e => JSON.parse(e))
class App extends Component {
  render() {
    return(
      <div className='App'>
        <Header />
        <Sidebar/>
        <Container />
      </div>
      
      )
  }
}

export default App;
