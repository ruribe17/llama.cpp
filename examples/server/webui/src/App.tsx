import { HashRouter, Outlet, Route, Routes } from 'react-router';
import Header from './components/Header';
import ConversationList from './components/ConversationList';
import { AppContextProvider } from './utils/app.context';
import ChatScreen from './components/ChatScreen';
import SettingDialog from './components/SettingDialog';

function App() {
  return (
    <HashRouter>
      <div className="flex flex-row drawer lg:drawer-open">
        <AppContextProvider>
          <Routes>
            <Route element={<AppLayout />}>
              <Route path="/chat/:convId" element={<ChatScreen />} />
              <Route path="*" element={<ChatScreen />} />
            </Route>
          </Routes>
        </AppContextProvider>
      </div>
    </HashRouter>
  );
}

function AppLayout() {
  return (
    <>
      <div className="flex w-full">
        <div
          id="convBlock"
          className="hidden lg:block w-64 min-w-64 h-screen overflow-y-auto overflow-x-clip bg-base-200"
        >
          <ConversationList />
        </div>
        <div
          id="mainBlock"
          className="block w-full min-w-0 h-screen overflow-y-auto overflow-x-clip"
        >
          <Header />
          <Outlet />
        </div>
        <div
          id="settingBlock"
          className="w-full hidden xl:block xl:max-w-md xl:min-w-md bg-base-200 overflow-y-auto overflow-x-clip "
        >
          <SettingDialog />
        </div>
        <button type="button" id="dropdown-close-helper" className="h-0 w-0" />
      </div>
    </>
  );
}

export default App;
